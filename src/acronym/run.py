import re
import pandas as pd


from src.acronym.task_init import AcronymGenTaskInit
from src.acronym.task_iterate import AcronymGenTaskIterate
from src.acronym.feedback import AcronymGenFeedback
from src.utils import retry_parse_fail_prone_cmd

CODEX = "code-davinci-002"
GPT3 = "text-davinci-003"
CHAT_GPT = "gpt-3.5-turbo"
GPT4 = "gpt-4"


ENGINE = CHAT_GPT

@retry_parse_fail_prone_cmd
def iterative_acronym(title: str, max_attempts: int) -> str:
    
    # initialize all the required components
    
    # generation of the first acronym
    task_init = AcronymGenTaskInit(engine=ENGINE, prompt_examples="data/prompt/acronym/init.jsonl")
    
    # getting feedback
    task_feedback = AcronymGenFeedback(engine=ENGINE, prompt_examples="data/prompt/acronym/feedback.jsonl")

    # iteratively improving the acronym
    task_iterate = AcronymGenTaskIterate(engine=ENGINE, prompt_examples="data/prompt/acronym/feedback.jsonl")
    
    
    # Initialize the task

    n_attempts = 0
    
    print(f"{n_attempts} INIT> {title}")
    acronyms_to_scores = dict()
    
    all_acronyms_to_scores = dict()
    best_score_so_far = 0
    while n_attempts < max_attempts:

        if n_attempts == 0:
            acronym = task_init(title=title)
        else:
            new_title, acronym = task_iterate(acronyms_to_scores=acronyms_to_scores)
            title = new_title

        
        scores = task_feedback(title=title, acronym=acronym)
        # extract expression "Total score: 22/25" from scores
        total_score = re.search(r"Total score: (\d+)/(\d+)", scores).group(0)
        total_score = int(total_score.split(":")[1].strip().split("/")[0])
        
        all_acronyms_to_scores[acronym] = {
            "scores": scores,
            "total_score": total_score,
            "title": title,
        }
        print(f"{n_attempts} GEN> {acronym} TITLE> {title}")

        print(f"{n_attempts} SCORES> {scores}")
        if total_score >= 0:  # only iterate over things that are improving
            best_score_so_far = total_score
            
            acronyms_to_scores[acronym] = (title, scores)
            
            
        else:
            print(f"Score of {acronym} is {total_score}, which is less than the current best of {best_score_so_far}")

        n_attempts += 1

    return all_acronyms_to_scores


def run_over_titles(titles_file: str, max_attempts: int, outfile: str):
    
    def _parse_results(title: str) -> str:
        try:
            results = iterative_acronym(title=title, max_attempts=max_attempts)
            if results is None:
                return "FAILED"
            res = []
            for acronym, scores in results.items():
                res.append(f"{acronym} [score: {scores['total_score']}] \n {scores['scores']}")
            return "\n ------ \n ".join(res)
        except Exception as e:
            return "FAILED"


    data = pd.read_csv(titles_file, sep="\t")
    data['generated_acronym'] = data['title'].apply(_parse_results)
        
    data.to_json(outfile, orient="records", lines=True)

if __name__ == "__main__":
    import sys
    title = sys.argv[1]  # Light Amplification by Stimulated Emission of Radiation
    if len(sys.argv) > 2:
        run_over_titles(titles_file=sys.argv[1], max_attempts=int(sys.argv[2]), outfile=sys.argv[3])
    else:
        max_attempts = 5
        all_acronyms_to_scores = iterative_acronym(
            title=title,
            max_attempts=max_attempts,
        )
        
        res = []
        for acronym, scores in all_acronyms_to_scores.items():
            res.append(f"{acronym} [score: {scores['total_score']}] \n {scores['scores']}")
        print("\n ------ \n ".join(res))

