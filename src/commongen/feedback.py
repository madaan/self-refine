import re
from typing import Set, List
import pandas as pd
from prompt_lib.backends import openai_api
import nltk
import spacy

nlp = spacy.load("en_core_web_sm")
from src.utils import Prompt


class CommongenFeedback(Prompt):
    def __init__(self, engine: str, prompt_examples: str, max_tokens: int = 300, temperature: float = 0.0) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_instr = "what concepts from the concept list are missing from the sentence and does the sentence make sense?"
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        template = """Concepts: {concepts}
Sentence: {sentence}
{prompt_instr}

Concept Feedback: {feedback}
Commonsense Feedback: {commonsense_feedback}"""

        examples_df = pd.read_json(examples_path, orient="records", lines=True)
        prompt = []
        for _, row in examples_df.iterrows():
            prompt.append(
                template.format(
                    concepts=row["concepts"],
                    sentence=row["sentence"],
                    feedback=", ".join(row["concept_feedback"]),
                    commonsense_feedback=row["commonsense_feedback"],
                    prompt_instr=self.prompt_instr,
                )
            )

        instruction = """We want to create a sentence that contains all the specified concepts. Please provide feedback on the following sentences. The feedback indicates missing concepts."""
        instruction = ""
        self.prompt = instruction + self.inter_example_sep.join(prompt)
        self.prompt = self.inter_example_sep.join(prompt) + self.inter_example_sep
    
    def __call__(self, sentence: str, concepts: List[str]):
        prompt = self.make_query(sentence=sentence, concepts=concepts)
        
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=prompt,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="###",
            temperature=self.temperature,
        )
        
        generated_feedback = openai_api.OpenaiAPIWrapper.get_first_response(output)

        concept_feedback = re.search(r"Concept Feedback: (.*)", generated_feedback)
        concept_feedback = concept_feedback.group(1) if concept_feedback else "NONE"
        # concept_feedback = self.fix_feedback(sentence=sentence, concepts=concepts, feedback=concept_feedback)
        
        commonsense_feedback = re.search(r"Commonsense Feedback: (.*)", generated_feedback)
        commonsense_feedback = commonsense_feedback.group(1) if commonsense_feedback else "NONE"

        return concept_feedback, commonsense_feedback


    def make_query(self, concepts: List[str], sentence: str):
        question = f"""Concepts: {concepts}
Sentence: {sentence}
{self.prompt_instr}"""
        return f"""{self.prompt}{question}"""
    
    
    def fix_feedback(self, sentence: str, concepts: List[str], feedback: str):
        """We rely on the model for generating a feedback. This is done to capture different forms in which the same concept might be expressed. However, the model might make mistakes and our task is simple enough that some of the mistakes can be corrected"""
        
        # print(f"Fixing feedback: {feedback}")
        concepts_in_sent = self.detect_concepts(sentence=sentence, concepts=concepts)
        missing_concepts = list(set(concepts).difference(concepts_in_sent))
        return ", ".join(missing_concepts)
        concepts_in_feedback = set([f.strip() for f in feedback.split(", ")])
        fixed_feedback = concepts_in_feedback.difference(concepts_in_sent)
        if len(fixed_feedback) == 0:
            return "None"
        return ", ".join(fixed_feedback)
        
    def detect_concepts(self, sentence: str, concepts: List[str]) -> Set[str]:
        present_concepts = []
        
        # Tokenize the sentence and lemmatize the tokens
        tokens = nltk.word_tokenize(sentence)
        lemmas = [token.lemma_ for token in nlp(sentence)]
        
        # Check if each concept is present in the sentence
        for concept in concepts:
            if concept in tokens or concept in lemmas:
                present_concepts.append(concept)
        
        return set(present_concepts)




if __name__ == "__main__":
    task_feedback = CommongenFeedback(
        prompt_examples="data/prompt/commongen/feedback.jsonl",
        engine="davinci-code-002"
    )
    
    print(task_feedback.prompt)
    
 