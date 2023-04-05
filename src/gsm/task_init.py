import pandas as pd
from src.utils import Prompt

from prompt_lib.backends import openai_api


class GSMInit(Prompt):
    def __init__(self, prompt_examples: str, engine: str, temperature: float) -> None:
        super().__init__(
            question_prefix="# Q: ",
            answer_prefix="# solution using Python:\n",
            intra_example_sep="\n",
            inter_example_sep="\n\n",
            engine=engine,
            temperature=temperature,
        )
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, prompt_examples) -> str:
        with open(prompt_examples, "r") as f:
            self.prompt = f.read()
    
    def make_query(self, solution: str) -> str:
        solution = solution.strip()
        query = f"{self.prompt}{self.question_prefix}{solution}{self.intra_example_sep}{self.answer_prefix}"
        return query

    def __call__(self, solution: str) -> str:
        generation_query = self.make_query(solution)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=300,
            stop_token=self.inter_example_sep,
            temperature=self.temperature,
        )

        solution_code = openai_api.OpenaiAPIWrapper.get_first_response(output)

        return solution_code.strip()


def test():
    task_init = GSMInit(
        prompt_examples="data/prompt/gsm/init.txt",
        engine="code-davinci-002",
        temperature=0.0,
    )

    question = "The educational shop is selling notebooks for $1.50 each and a ballpen at $0.5 each.  William bought five notebooks and a ballpen. How much did he spend in all?"
    print(task_init(question))
    

if __name__ == "__main__":
    test()