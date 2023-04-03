import sys
from typing import Dict, List
from src.utils import Prompt
import pandas as pd

from prompt_lib.backends import openai_api


class AcronymGenTaskIterate(Prompt):
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
        
        prompt_examples = pd.read_json(prompt_examples, orient="records", lines=True)
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
        
        instr = """We want to iteratively improve acronyms. To help improve, scores for each acronym on five desired traits are provided: i) ease of pronunciation, ii) ease of spelling, and iii) relation to the title, iv) positive connotation, v) well-known.

"""
        example_template = """Title: {title}

Acronym: {acronym}

Scores:

* Ease of pronunciation: {pronunciation_score}
* Ease of spelling: {spelling_score}
* Relation to title: {relation_score}
* Positive connotation: {connotation_score}
* Well-known: {well_known_score}

* Total score: {total_score}

Okay, let's use this feedback to improve the acronym.

"""     
        prompt = []
        for example in incrementally_improving_examples:
            prompt.append(example_template.format(**example))
        
        prompt = "".join(prompt)
        prompt = instr + prompt
        return prompt.strip()

    def make_query(self, question: str) -> str:
        return f"{self.prompt}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"
        # return super().make_query(prompt, question)

    def _make_input(
        self,
        title: str,
        acronym: str,
        scores: str,
    ) -> str:
        input_txt = f"""Title: {title}

Acronym: {acronym}

Scores:

{scores}

Okay, let's use this feedback to improve the acronym.

"""

        return input_txt

    def __call__(
        self,
        acronyms_to_scores: Dict[str, str],
    ) -> str:
        example_input = self.make_input(
            acronyms_to_scores=acronyms_to_scores,
        )
        transfer_query = self.make_query(example_input)
        self.count += 1
        with open(f"acronym_iterate_{self.count}.txt", "w") as f:
            f.write(transfer_query + "\n")

        output = openai_api.OpenaiAPIWrapper.call(
            prompt=transfer_query,
            engine=self.engine,
            max_tokens=300,
            stop_token=self.inter_example_sep,
            temperature=0.7,
        )
        response = openai_api.OpenaiAPIWrapper.get_first_response(output)

        
        acronym = response.split("Acronym:")[1].strip().split("\n")[0].strip()
        
        new_title = response.split("Title:")[1].strip().split("\n")[0].strip()
        
        
        
        return new_title, acronym.strip()

    def make_input(
        self,
        acronyms_to_scores: Dict[str, str],
    ) -> str:
        input_txt = ""
        for acronym, (title, scores) in acronyms_to_scores.items():
            input_txt += self._make_input(
                title=title,
                acronym=acronym,
                scores=scores,
            )
        return input_txt




    

if __name__ == "__main__":
    obj = AcronymGenTaskIterate(prompt_examples="data/prompt/acronym/feedback.jsonl", engine="whatever")
    print(obj.prompt)