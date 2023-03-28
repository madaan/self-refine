import pandas as pd
from src.utils import Prompt
from typing import List, Optional, Union
import sys
from prompt_lib.backends import openai_api


class ResponseGenTaskInit(Prompt):
    def __init__(self, prompt_examples: str, engine: str, numexamples=3) -> None:
        super().__init__(
            question_prefix="Conversation history: ",
            answer_prefix="Response: ",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.setup_prompt_from_examples_file(prompt_examples, numexamples=numexamples)

    def setup_prompt_from_examples_file(self, examples_path: str, numexamples=10) -> str:
        instruction = (
            "Provided a dialogue between two speakers, generate a response that is coherent with the dialogue history. Desired traits for responses are: 1) Relevant - The response addresses the context, 2) Informative - The response provides some information, 3) Interesting - The response is not interesting, 4) Consistent - The response is consistent with the rest of the conversation in terms of tone and topic, 5) Helpful - The response is helpful in providing any information or suggesting any actions, 6) Engaging - The response is not very engaging and does not encourage further conversation, 7) Specific - The response contains pecific content, 9) User understanding - The response demonstrates an understanding of the user's input and state of mind, and 10) Fluent. Response should begin with - Response:\n\n"
        )

        examples_df = pd.read_json(examples_path, orient="records", lines=True)
        prompt = []
        for i, row in examples_df.iterrows():
            if i >= numexamples:
                break
            prompt.append(self._build_query_from_example(row["history"], row["response"]))

        self.prompt = instruction + self.inter_example_sep.join(prompt) + self.inter_example_sep

    def _build_query_from_example(self, history: Union[str, List[str]], response: Optional[str]=None) -> str:
        history = history.replace('System: ', '').replace('User: ', '')

        TEMPLATE = """Conversation history: 

{history}

Response: {response}"""
            
        query = TEMPLATE.format(history=history, response=response)
        return query

    def make_query(self, context: str) -> str:
        context = context.replace('System: ', '').replace('User: ', '')
        query = f"{self.prompt}{self.question_prefix}\n\n{context}{self.intra_example_sep}"
        return query

    def __call__(self, context: str) -> str:
        generation_query = self.make_query(context)
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=800,
            stop_token="###",
            temperature=0.7,
        )

        generated_response = openai_api.OpenaiAPIWrapper.get_first_response(output)

        generated_response = generated_response.split(self.answer_prefix)[1].replace("#", "").strip()


        return output, generated_response.strip()
