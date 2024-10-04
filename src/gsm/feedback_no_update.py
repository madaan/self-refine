import pandas as pd
from prompt_lib.backends import openai_api

from src.utils import Prompt


class GSMFeedback(Prompt):
    def __init__(self, engine: str, prompt_examples: str, temperature: float, max_tokens: int = 300) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n### END ###\n\n",
            engine = engine,
            temperature = temperature
        )
        self.max_tokens = max_tokens
        self.instruction = "# There is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good."
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        with open(examples_path, "r") as f:
            self.prompt = f.read()
    
    def __call__(self, solution: str):
        generation_query = self.make_query(solution=solution)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="### END",
            temperature=self.temperature,
        )
        
        entire_output = openai_api.OpenaiAPIWrapper.get_first_response(output)
        if "### END" in entire_output:
            entire_output = entire_output.split("### END")[0]
        solution = entire_output.split("def solution():")[1]
        feedback = entire_output.split("def solution():")[0]
        solution = "def solution():" + solution.rstrip()
        return {"solution": solution, "feedback": feedback}

    def make_query(self, solution: str):
        solution = f"""{self.question_prefix}{solution}{self.intra_example_sep}{self.instruction}{self.answer_prefix}"""
        return f"{self.prompt}{solution}"
    

def test():
    task_fb = GSMFeedback(
        prompt_examples="data/prompt/gsm/feedback.txt",
        engine="code-davinci-002",
        temperature=0.7,
    )

    wrong_soln = """def solution():
    \"\"\"Twenty dozen cups cost $1200 less than the total cost of half a dozen plates sold at $6000 each. Calculate the total cost of buying each cup.\"\"\"
    plates = 6
    plate_cost = 6000
    cups = 12 * 20
    cup_cost = (plates * plate_cost) / cups - 1200
    result = cup_cost
    return result"""
    feedback_and_solution = task_fb(wrong_soln)
    print(feedback_and_solution["feedback"])
    print(feedback_and_solution["solution"])
    

if __name__ == '__main__':
    test()
