import sys
from typing import Dict, List
from src.utils import Prompt

from prompt_lib.backends import openai_api


class ResponseGenTaskIterate(Prompt):
    def __init__(self, engine: str, prompt_examples: str) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.count = 0
        self.prompt = self.make_prompt(prompt_examples=prompt_examples)

    def make_prompt(self, prompt_examples: str, reduce_window=0) -> str:
        import pandas as pd
        prompt_examples = pd.read_json(prompt_examples, orient="records")
        prompt_examples = prompt_examples[reduce_window:]
        # group on example
        grouped = prompt_examples.groupby("example")
        
        prompt = []
        # sort each group by score
        for _, group in grouped:
            group["numerical_score"] = group["total_score"].apply(lambda x: int(x.split("/")[0].strip()))
            group = group.sort_values("numerical_score")
            prompt.append(self.make_one_iterate_example(group.to_dict("records")))
        
        return self.inter_example_sep.join(prompt) + self.inter_example_sep
        

    def make_one_iterate_example(self, incrementally_improving_examples: List[Dict]):
        """Given a list of examples that are incrementally improving, return a new example.
        """
        
        instr = """We want to iteratively improve the provided responses. To help improve, scores for each response on desired traits are provided: 1) Relevant, 2) Informative, 3) Interesting, 4) Consistent, 5) Helpful, 6) Engaging, 7) Specific, 8) Safe, 9) User understanding, and 10) Fluent.

"""
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

* Total score: {total_score}

Okay, let's use this feedback to improve the response.

"""     
        prompt = []
        for row in incrementally_improving_examples:
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

        
        prompt = "".join(prompt)
        prompt = instr + prompt
        return prompt.strip()

    def make_query(self, question: str, reduce_window=0) -> str:
        if reduce_window>0:
            self.prompt = self.make_prompt(prompt_examples="data/prompt/responsegen/feedback.jsonl", reduce_window=reduce_window)
        question = question.replace('System: ', '').replace('User: ', '')
        return f"{self.prompt}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"
        # return super().make_query(prompt, question)

    def _make_input(
        self,
        context: str,
        response: str,
        scores: str,
    ) -> str:
        context = context.replace('System: ', '').replace('User: ', '')
        input_txt = f"""Conversation history: 
        
{context}

Response: {response}

Scores:

{scores}

Okay, let's use this feedback to improve the response.

Conversation history: 
        
{context}
"""

        return input_txt

    def __call__(
        self,
        responses_to_scores: Dict[str, str],
        reduce_window=0
    ) -> str:
        example_input = self.make_input(
            responses_to_scores=responses_to_scores
        )
        transfer_query = self.make_query(example_input, reduce_window=reduce_window)
        self.count += 1
        with open(f"responses_iterate_{self.count}.txt", "w") as f:
            f.write(transfer_query + "\n")
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=transfer_query,
            engine=self.engine,
            max_tokens=200,
            stop_token=self.inter_example_sep,
            temperature=0.7,
        )
        modelresponse = openai_api.OpenaiAPIWrapper.get_first_response(output)
        response = modelresponse.split("Response:")[1].strip().split("\n")[0].strip()

        
        return output, response.strip()

    def make_input(
        self,
        responses_to_scores: Dict[str, str],
    ) -> str:
        input_txt = ""
        for response, (context, scores) in responses_to_scores.items():
            context = context.replace('System: ', '').replace('User: ', '')
            input_txt += self._make_input(
                context=context,
                response=response,
                scores=scores,
            )
        return input_txt




    

if __name__ == "__main__":
    obj = ResponseGenTaskIterate(prompt_examples="data/prompt/acronym/feedback.v2.jsonl", engine="whatever")
    print(obj.prompt)