import re
from typing import Dict, List
from src.utils import Prompt

from prompt_lib.backends import openai_api

header = """Concepts: {concepts}
"""
example_template = """Sentence: {sentence}

what concepts from the concept list are missing from the sentence?

Concept Feedback: {concept_feedback}

Any feedback on commonsense?

Commonsense Feedback: {commonsense_feedback}"""

instr = """

Okay, impove the sentence using the feedback:

"""

class CommongenTaskIterate(Prompt):
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

    def make_prompt(self, prompt_examples: str) -> str:
        import pandas as pd

        prompt_examples = pd.read_json(prompt_examples, orient="records", lines=True)

        prompt = []

        for example in prompt_examples.to_dict(orient="records"):
            prompt.append(
                self.make_one_iterate_example(
                    concepts=example["concepts"], sent_to_fb=example["sentence_to_feedback"]
                )
            )

        return self.inter_example_sep.join(prompt) + self.inter_example_sep

    def make_one_iterate_example(self, concepts: List[str], sent_to_fb: List[Dict]):
        """Given a list of examples that are incrementally improving, return a new example."""



        single_example = []
        for example in sent_to_fb:

            single_example.append(example_template.format(
                sentence=example["sentence"], commonsense_feedback=example["commonsense_feedback"], concept_feedback=example["concept_feedback"]
            ))

        return header.format(concepts=concepts) + instr.join(single_example)

    def make_query(self, concepts: List[str],
        sent_to_fb: List[Dict],) -> str:
        query_example = self.make_one_iterate_example(concepts=concepts, sent_to_fb=sent_to_fb)
        return f"{self.prompt}{self.question_prefix}{query_example}{self.intra_example_sep}{self.answer_prefix}" + instr
        # return super().make_query(prompt, question)

    def __call__(
        self,
        concepts: List[str],
        sent_to_fb: List[Dict],
    ) -> str:
    
        transfer_query = self.make_query(concepts=concepts, sent_to_fb=sent_to_fb)
        self.count += 1

        output = openai_api.OpenaiAPIWrapper.call(
            prompt=transfer_query,
            engine=self.engine,
            max_tokens=300,
            stop_token=self.inter_example_sep,
            temperature=0.7,
        )
        response = openai_api.OpenaiAPIWrapper.get_first_response(output)

        print("######")
        print()
        print("------")
        print(response)
        print("------")

        response = re.search("Sentence: (.*)", response).group(1).strip().split("\n")[0].strip()

        return response.strip()

    def make_input(
        self,
        title: str,
        acronyms_to_scores: Dict[str, str],
    ) -> str:
        input_txt = ""
        for acronym, scores in acronyms_to_scores.items():
            input_txt += self._make_input(
                title=title,
                acronym=acronym,
                scores=scores,
            )
        return input_txt


if __name__ == "__main__":
    obj = CommongenTaskIterate(
        prompt_examples="data/prompt/commongen/iterate.v1.jsonl", engine="whatever"
    )
    print(obj.prompt)
    # print(obj.make_query(concepts=["a", "b"], sent_to_fb=[{"sentence": "a", "feedback": "a"}, {"sentence": "b", "feedback": "d"}]))
