import pandas as pd
from prompt_lib.backends import openai_api

from src.utils import Prompt


class ResponseGenFeedback(Prompt):
    def __init__(self, engine: str, prompt_examples: str, max_tokens: int = 400) -> None:
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
        template = """Conversation history: 
        
{history}

Response: {response}

Scores:

* Relevant: {Relevant}
* Informative: {Informative}
* Interesting: {Interesting}
* Consistent: {Consistent}
* Helpful: {Helpful}
* Engaging : {Engaging}
* Specific: {Specific}
* Safe: {Safe}
* User understanding: {Userunderstanding}
* Fluent: {Fluent}
* Total score: {total_score}"""
        examples_df = pd.read_json(examples_path, orient="records")
        prompt = []
        for _, row in examples_df.iterrows():
            prompt.append(
                template.format(
                    history=row['history'].replace('System: ', '').replace('User: ', ''),
                    response=row["response"],
                    Relevant=row["Relevant"],
                    Informative=row["Informative"],
                    Interesting=row["Interesting"],
                    Consistent=row["Consistent"],
                    Helpful=row["Helpful"],
                    Engaging=row["Engaging"],
                    Specific=row["Specific"],
                    Safe=row["Safe"],
                    Userunderstanding=row["Userunderstanding"],
                    Fluent=row["Fluent"],
                    total_score=row["total_score"],
                )
            )

        instruction = """We want to iteratively improve the provided responses. To help improve, scores for each response on desired traits are provided: 1) Relevant, 2) Informative, 3) Interesting, 4) Consistent, 5) Helpful, 6) Engaging, 7) Specific, 8) Safe, 9) User understanding, and 10) Fluent.

Here are some examples of this scoring rubric:

"""
        self.prompt = instruction + self.inter_example_sep.join(prompt)
        self.prompt = self.inter_example_sep.join(prompt) + self.inter_example_sep
    
    def __call__(self, context: str, response: str):
        prompt = self.get_prompt_with_question(context=context, response=response)

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

        return output, generated_feedback

    def get_prompt_with_question(self, context: str, response: str):
        context = context.replace('System: ', '').replace('User: ', '')
        question = self.make_query(context=context, response=response)
        return f"""{self.prompt}{question}\n\n"""

    def make_query(self, context: str, response: str):
        question = f"""Conversation history: 
        
{context}

Response: {response}"""
        return question
