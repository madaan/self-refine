import re
import math


from src.acronym.task_init import AcronymGenTaskInit
from src.acronym.task_iterate import AcronymGenTaskIterate
from src.acronym.feedback import AcronymGenFeedback
from src.utils import retry_parse_fail_prone_cmd

CODEX = "code-davinci-002"
GPT3 = "text-davinci-003"
CHAT_GPT = "gpt-3.5-turbo"
GPT4 = "gpt-4"
ENGINE = CHAT_GPT

task_init = AcronymGenTaskInit(engine=ENGINE, prompt_examples="data/prompt/acronym/init.jsonl")

# getting feedback
task_feedback = AcronymGenFeedback(
    engine=ENGINE, prompt_examples="data/prompt/acronym/feedback.jsonl"
)

# iteratively improving the acronym
task_iterate = AcronymGenTaskIterate(
    engine=ENGINE, prompt_examples="data/prompt/acronym/feedback.jsonl"
)


class TreeNode:
    def __init__(self, title: str, acronym: str, scores: dict, parent=None):
        self.title = title
        self.acronym = acronym.strip()
        self.scores = scores
        self.children = []
        self.visits = 1
        self.value = 0.0
        self.parent = parent

    def __str__(self):
        children_str = ", ".join([str(child.acronym) for child in self.children])
        if children_str == "":
            children_str = "None"
        res = f"TreeNode(title='{self.title}', acronym='{self.acronym}', scores={self.scores}, visits={self.visits}, value={self.value}, parent_acronym='{self.parent.acronym if self.parent else None}' children={children_str})"
        return res


@retry_parse_fail_prone_cmd
def generate_initial_children(node: TreeNode, task_iterate, task_feedback, num_children: int = 3):
    for _ in range(num_children):
        new_title, new_acronym = task_iterate(
            acronyms_to_scores={node.acronym: (node.title, node.scores)}
        )
        child = TreeNode(title=new_title, acronym=new_acronym, scores=None, parent=node)
        simulate(child, task_feedback)  # Set scores for the child node
        node.children.append(child)


def parse_scores(scores_output: str) -> dict:
    scores_pattern = re.compile(r"\* (.*?): (.*)\n?")
    scores = {}

    for score_match in scores_pattern.finditer(scores_output):
        score_title, score_value = score_match.groups()
        score_value = int(re.search(r"\d+", score_value).group(0))  # Updated
        scores[score_title] = score_value

    return scores


def normalize_scores(scores: dict) -> dict:
    normalized_scores = {}
    for key, value in scores.items():
        if key != "Total score":
            normalized_scores[key] = value / 5
    return normalized_scores


def weighted_sum(normalized_scores: dict, weights: dict) -> float:
    return sum(normalized_scores[key] * weights[key] for key in normalized_scores)


def select(node: TreeNode, weights: dict, C: float = 1.0):
    if not node.children:
        return node

    ucb1_values = [
        (weighted_sum(normalize_scores(child.scores), weights) / child.visits)
        + C * math.sqrt(math.log(node.visits) / child.visits)
        for child in node.children
    ]

    max_index = ucb1_values.index(max(ucb1_values))
    return select(node.children[max_index], weights, C)


@retry_parse_fail_prone_cmd
def expand(node: TreeNode, task_iterate, expanded_nodes_cache: set):
    acronyms_to_scores = {}
    current = node
    while current is not None:
        acronyms_to_scores[current.acronym] = (current.title, current.scores)
        current = current.parent

    new_title, new_acronym = task_iterate(acronyms_to_scores=acronyms_to_scores)

    while new_acronym in expanded_nodes_cache:
        new_title, new_acronym = task_iterate(acronyms_to_scores=acronyms_to_scores)

    expanded_nodes_cache.add(new_acronym)
    return TreeNode(title=new_title, acronym=new_acronym, scores=None, parent=node)


@retry_parse_fail_prone_cmd
def simulate(node: TreeNode, task_feedback):
    scores = task_feedback(title=node.title, acronym=node.acronym)
    node.scores = parse_scores(scores)
    normalized_total_score = node.scores["Total score"] / 25
    return normalized_total_score


def backpropagate(node: TreeNode, value: float):
    node.visits += 1
    node.value += value

    if node.parent is not None:
        backpropagate(node.parent, value)


def mcts_iteration(
    root: TreeNode, weights: dict, task_iterate, task_feedback, expanded_nodes_cache: set
):

    print("Selecting...")
    selected_node = select(root, weights)
    print(f"  Selected node: {selected_node.acronym} for title '{selected_node.title}'")

    print("Expanding...")
    expanded_node = expand(selected_node, task_iterate, expanded_nodes_cache)

    selected_node.children.append(
        expanded_node
    )  # Add expanded node as a child of the selected node
    print(f"  Expanded node: {expanded_node.acronym} for title '{expanded_node.title}'")

    print("Simulating...")
    value = simulate(expanded_node, task_feedback)
    print(f"  Simulated value: {value}")

    print("Backpropagating...")
    backpropagate(expanded_node, value)
    print("  Backpropagation complete")


root_title = "Iterative Refinement with Self-Feedback"
root_acronym = task_init(title=root_title)
root_scores = task_feedback(title=root_title, acronym=root_acronym)
root_scores = parse_scores(root_scores)
root = TreeNode(root_title, root_acronym, scores=root_scores)

print(f"Root node: {root}")

generate_initial_children(root, task_iterate, task_feedback)

print(f"Root node after generating initial children: {root}")

weights = {
    "Ease of pronunciation": 0.2,
    "Ease of spelling": 0.2,
    "Relation to title": 0.3,
    "Positive connotation": 0.2,
    "Well-known": 0.1,
}

iterations = 4
expanded_nodes_cache = {root.acronym}

for _ in range(iterations):
    mcts_iteration(root, weights, task_iterate, task_feedback, expanded_nodes_cache)


def dfs(node: TreeNode, best_node: TreeNode):
    if not node.children:
        if node.scores["Total score"] > best_node.scores["Total score"]:
            return node
        else:
            return best_node

    for child in node.children:
        best_node = dfs(child, best_node)

    return best_node


best_node = dfs(root, root)
best_acronym, best_score = best_node.acronym, best_node.scores["Total score"]

print(f"\nBest acronym: {best_acronym} with total score: {best_score}")


def print_tree(node: TreeNode, indent: int = 0):
    indentation = "  " * indent
    print(f"{indentation}{node.acronym} (Score: {node.scores['Total score']})")

    for child in node.children:
        print_tree(child, indent + 1)


print_tree(root)
