import pandas as pd
from src.utils import Prompt

from prompt_lib.backends import openai_api


class AcronymGenTaskInit(Prompt):
    def __init__(self, prompt_examples: str, engine: str) -> None:
        super().__init__(
            question_prefix="Title: ",
            answer_prefix="Acronym: ",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        TEMPLATE = """Title: {title}

Acronym: {answer}"""

        examples_df = pd.read_json(examples_path, orient="records", lines=True)
        prompt = []
        for _, row in examples_df.iterrows():
            prompt.append(TEMPLATE.format(title=row["title"], answer=row["acronym"]))
        self.prompt = self.inter_example_sep.join(prompt)
        self.prompt = self.prompt + self.inter_example_sep
    
    def make_query(self, title: str) -> str:
        query = f"{self.prompt}{self.question_prefix}{title}{self.intra_example_sep}"
        return query

    def __call__(self, title: str) -> str:
        generation_query = self.make_query(title)

        output = openai_api.OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=300,
            stop_token="###",
            temperature=0.7,
        )

        generated_acronym = openai_api.OpenaiAPIWrapper.get_first_response(output)
        # print("output:")
        # print(generated_acronym)
        # sys.exit()
        generated_acronym = generated_acronym.split(self.answer_prefix)[1].replace("#", "").strip()
        return generated_acronym.strip()


def test():
    task_init = AcronymGenTaskInit(engine="code-davinci-002", prompt_examples="data/prompt/acronym/init.jsonl")
    print(task_init.prompt)


if __name__ == "__main__":
    test()