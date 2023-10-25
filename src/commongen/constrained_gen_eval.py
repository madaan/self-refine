import openai
import sys
from typing import List
import tiktoken
import time
from tqdm import tqdm
import pandas as pd
import random
random.seed(42)



def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,openai.error.ServiceUnavailableError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:

                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                print(f"Retrying in {delay} seconds.")

                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper



@retry_with_exponential_backoff
def call_openai_api(start_chat_log, stop_token, max_tokens=300, temperature=0.0, engine="gpt-4-0613"):
    response = openai.ChatCompletion.create(
      model=engine,
      messages=start_chat_log,
      max_tokens=max_tokens,
      temperature=temperature,
      stop=stop_token
    )
    return response['choices'][0]['message']['content']

class ChatGPTWrapper:
    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
        # from openai-cookbook
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    @staticmethod
    def score_story(story_a: str, story_b: str) -> str:
        # perform A/B switch, so that the order of the stories is not always the same
        flipped = False
        if random.random() < 0.5:
            story_a, story_b = story_b, story_a
            flipped = True

        response = call_openai_api(
            start_chat_log=[
                {
                    "role": "system",
                    "content": "You are an expert in storytelling. Pick the better story.",
                },
                {
                    "role": "user",
                    "content": f"""Which story is better?
Story A: {story_a}

Story B: {story_b}.

Judge the story based on the flow, the grammar, and the overall quality of the story. Rate more realistic story higher. Pick your answer from ['Story A', 'Story B', 'either']. First, reason about your choice. Then, generate 'The better story is Story A' or 'The better story is Story B' or 'The better story is either'.

Format:

Reasoning: <your reasoning>. The better story is <your choice>.


Reasoning:""",
                },
            ],
            max_tokens=200,
            engine="gpt-4-0613",
            temperature=0.0,
            stop_token="###\n\nDemonstration",
        )

        if flipped:  # change all As to Bs and vice versa, it'll add typos but we care about the preference.
            response = response.replace("Story B", "Story Z")
            response = response.replace("Story A", "Story B")
            response = response.replace("Story Z", "Story A")

        # print(response)
        return response

from collections import Counter
import re


import pandas as pd
import sys
from typing import List, Set
from tqdm import tqdm
import spacy
import nltk
import random
from collections import Counter

# Initialize SpaCy and NLTK
nlp = spacy.load('en_core_web_sm')

# Function to detect concepts in a sentence
def detect_concepts(sentence: str, concepts: List[str]) -> Set[str]:
    present_concepts = set()
    sentence = sentence.lower()
    nltk_tokens = set(nltk.word_tokenize(sentence))
    spacy_doc = nlp(sentence)
    spacy_lemmas = set(token.lemma_ for token in spacy_doc)
    all_tokens = nltk_tokens.union(spacy_lemmas)
    for concept in concepts:
        concept_lower = concept.lower()
        if concept_lower in all_tokens:
            present_concepts.add(concept)
    return present_concepts

# Function to calculate missing concepts percentage
def calculate_missing_concepts_percentage(sentence: str, concepts: List[str]) -> float:
    concepts = set(concepts)
    present_concepts = detect_concepts(sentence, concepts)
    absent_concepts = concepts - present_concepts
    missing_percentage = (len(absent_concepts) / len(concepts)) * 100
    return missing_percentage

# Main function
def main(jsonl_filepath: str):
    df = pd.read_json(jsonl_filepath, lines=True, orient='records')
    results = []
    win_counts = Counter()

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if not isinstance(row["sent_to_fb"], list) or len(row["sent_to_fb"]) < 2:
            continue
        first_sentence = row["sent_to_fb"][0]["sentence"]
        last_sentence = row["sent_to_fb"][-1]["sentence"]
        comparison_result = ChatGPTWrapper.score_story(first_sentence, last_sentence)
        choice_match = re.search(r'The better story is (Story A|Story B|either)', comparison_result)
        if choice_match:
            choice = choice_match.group(1)
        else:
            choice = "Unknown"
            
        first_missing = calculate_missing_concepts_percentage(first_sentence, row['concepts'])
        last_missing = calculate_missing_concepts_percentage(last_sentence, row['concepts'])
        
        winner = None
        if choice == "Story A" and first_missing <= last_missing:
            winner = "Direct"
        elif choice == "Story B" and last_missing <= first_missing:
            winner = "Self-Refine"
        elif choice == "either" and first_missing == last_missing:
            winner = "Either"
        else:
            winner = "Neither"
        
        results.append(winner)
        win_counts[winner] += 1
        
        # Live win rates
        print(f"\nLive Win Rates after {idx + 1} comparisons:")
        total_comparisons = sum(win_counts.values())
        for k, v in win_counts.items():
            win_rate = v / total_comparisons * 100
            print(f"{k}: {win_rate:.2f}%")
        print()

    # Final win rates
    print("\nFinal Win Rates:")
    total_comparisons = sum(win_counts.values())
    for k, v in win_counts.items():
        win_rate = v / total_comparisons * 100
        print(f"{k}: {win_rate:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_jsonl_file>")
        sys.exit(1)
        
    jsonl_filepath = sys.argv[1]
    main(jsonl_filepath)
