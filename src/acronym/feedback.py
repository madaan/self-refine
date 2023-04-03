import pandas as pd
from prompt_lib.backends import openai_api

from src.utils import Prompt


class AcronymGenFeedback(Prompt):
    def __init__(self, engine: str, prompt_examples: str, max_tokens: int = 300) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.max_tokens = max_tokens
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        template = """Title: {title}

Acronym: {answer}

Scores:

* Ease of pronunciation: {pronunciation_score}
* Ease of spelling: {spelling_score}
* Relation to title: {relation_score}
* Positive connotation: {connotation_score}
* Well-known: {well_known_score}

* Total score: {total_score}"""

        examples_df = pd.read_json(examples_path, orient="records", lines=True)
        prompt = []
        for _, row in examples_df.iterrows():
            prompt.append(
                template.format(
                    title=row["title"],
                    answer=row["acronym"],
                    pronunciation_score=row["pronunciation_score"],
                    spelling_score=row["spelling_score"],
                    relation_score=row["relation_score"],
                    connotation_score=row["connotation_score"],
                    well_known_score=row["well_known_score"],
                    total_score=row["total_score"],
                )
            )

        instruction = """We want to score each acronym on five qualities: i) ease of pronunciation, ii) ease of spelling, and iii) relation to the title, iv) positive connotation, v) well-known.

Here are some examples of this scoring rubric:

"""
        self.prompt = instruction + self.inter_example_sep.join(prompt)
        self.prompt = self.inter_example_sep.join(prompt) + self.inter_example_sep
    
    def __call__(self, title: str, acronym: str):
        prompt = self.get_prompt_with_question(title=title, acronym=acronym)

        output = openai_api.OpenaiAPIWrapper.call(
            prompt=prompt,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="###",
            temperature=0.7,
        )
        
        generated_feedback = openai_api.OpenaiAPIWrapper.get_first_response(output)
        generated_feedback = generated_feedback.split("Scores:")[1].strip()
        generated_feedback = generated_feedback.split("#")[0].strip()
        return generated_feedback

    def get_prompt_with_question(self, title: str, acronym: str):
        question = self.make_query(title=title, acronym=acronym)
        return f"""{self.prompt}{question}"""

    def make_query(self, title: str, acronym: str):
        question = f"""Title: {title}

Acronym: {acronym}"""
        return question



if __name__ == "__main__":
    feedback = AcronymGenFeedback(
        engine="davinci-code-002",
        prompt_examples="data/prompt/acronym/feedback.jsonl",
    )
    
    print(feedback.prompt)