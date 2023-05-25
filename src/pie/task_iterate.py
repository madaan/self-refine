import sys
from typing import Dict, List
from src.utils import Prompt

from prompt_lib.backends import openai_api


class PieIterate(Prompt):
    def __init__(self, engine: str, prompt_examples: str, temperature: float, feedback_type: str = "default") -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="# Improved version:\n",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n### END ###n\n",
        )
        self.engine = engine
        self.count = 0
        self.temperature = temperature
        self.make_prompt(prompt_examples=prompt_examples)
        self.feedback_type = feedback_type

    def make_prompt(self, prompt_examples: str) -> str:
        with open(prompt_examples, "r") as f:
            self.prompt= f.read()

        # return super().make_query(prompt, question)

    def __call__(
        self,
        slow_code: str,
        feedback: str,
    ) -> str:
        generation_query = self.make_query(slow_code=slow_code, feedback=feedback)

        output = openai_api.OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=300,
            stop_token="### END",
            temperature=self.temperature,
        )
        generated_code = openai_api.OpenaiAPIWrapper.get_first_response(output)

        
        if "### END" in generated_code:
            generated_code = generated_code.split("### END")[0]
        return generated_code.strip()



    def make_query(self, slow_code: str, feedback: str) -> str:
        instr = "# Why is this code slow?" if self.feedback_type == "default" else "# How to improve this code?"
        example_template = """{slow_code}

{instr}

{feedback}

# Improved version:

"""     
        query = example_template.format(slow_code=slow_code, feedback=feedback)

        return f"{self.prompt}{query}"


def test():
    task_iterate = PieIterate(
        prompt_examples="data/prompt/pie/iterate.txt",
        engine="gpt-3.5-turbo",
        temperature=0.6
    )

    slow_code = "def sum(n):\\n    res = 0\\n    for i in range(n):\\n        res += i\\n    return res"
    feedback = "# This code is slow because it is using a brute force approach to calculate the sum of numbers up to n. It is looping through every number from 0 to n and adding it to the sum. This is a very slow approach because it has to loop through so many numbers. A better approach would be to use the formula for the sum of the numbers from 0 to n, which is (n(n+1))/2. Using this formula, you can calculate the sum of the numbers in constant time."
    # print(task_iterate.prompt)
    print(task_iterate(slow_code, feedback))
    
if __name__ == '__main__':
    test()