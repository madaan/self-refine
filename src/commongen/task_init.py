from typing import List
import pandas as pd
from src.utils import Prompt

from prompt_lib.backends import openai_api


class CommongenTaskInit(Prompt):
    def __init__(self, prompt_examples: str, engine: str, temperature: float = 0.0) -> None:
        super().__init__(
            question_prefix="Concepts: ",
            answer_prefix="Sentence: ",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.temperature = temperature
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        TEMPLATE = """Concepts: {concepts}

Sentence: {sentence}"""

        examples_df = pd.read_json(examples_path, orient="records", lines=True)
        prompt = []
        for _, row in examples_df.iterrows():
            prompt.append(TEMPLATE.format(concepts=row["concepts"], sentence=row["target"]))
        self.prompt = self.inter_example_sep.join(prompt)
        self.prompt = self.prompt + self.inter_example_sep
    
    def make_query(self, concepts: List[str]) -> str:
        
        query = f"""{self.question_prefix}{concepts}"""
        query = f"{self.prompt}{query}{self.intra_example_sep}"
        return query

    def __call__(self, concepts: List[str]) -> str:
        generation_query = self.make_query(concepts)
        
        generation_query = f"""{self.make_query(concepts)}
Do your best! It's okay if the sentence is not coherent.

Sentence:"""
        
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=300,
            stop_token="###",
            temperature=self.temperature,
        )

        generated_sent = openai_api.OpenaiAPIWrapper.get_first_response(output)
        if self.answer_prefix in generated_sent:
            generated_sent = generated_sent.split(self.answer_prefix)[1].strip()
        
        if "#" in generated_sent:
            generated_sent = generated_sent.replace("#", "").strip()

        return generated_sent.strip()



if __name__ == "__main__":
    task_init = CommongenTaskInit(
        prompt_examples="data/prompt/commongen/init.jsonl",
        engine="davinci-code-002"
    )
    
    print(task_init.prompt)
    # print(task_init.make_query(["a", "b", "c"]))