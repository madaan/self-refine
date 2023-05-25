from collections import Counter
import pandas as pd


import pandas as pd
import json

import pandas as pd
from typing import List, Dict
import tiktoken
import random
from prompt_lib.backends import openai_api
from tqdm import tqdm

random.seed(42)


class ChatGPTWrapper:
    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
        # from openai-cookbook
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    @staticmethod
    def score_review(review_a: str, review_b: str, target_sentiment: str) -> str:
        # perform A/B switch, so that the order of the stories is not always the same
        flipped = False
        if random.random() < 0.5:
            review_a, review_b = review_b, review_a
            flipped = True

        response = openai_api.OpenaiAPIWrapper.call(
            [
                {
                    "role": "system",
                    "content": "You are an expert in writing reviews. Pick the review that is more closely aligned with the target sentiment.",
                },
                {
                    "role": "user",
                    "content": f"""Which review is aligned with the sentiment {target_sentiment}?
Review A: {review_a}


Review B: {review_b}.

Pick your answer from ['Review A', 'Review B', 'both', 'neither']. Generate a short explanation for your choice first. Then, generate 'The more aligned review is A' or 'The more aligned review is B' or 'The more aligned review is both' or 'The more aligned review is neither'.

Format: <explanation> <answer> STOP

""",
                },
            ],
            max_tokens=100,
            engine="gpt-4",
            temperature=0.0,
            num_completions=1,
            stop_token="STOP",
        )
        response = openai_api.OpenaiAPIWrapper.get_first_response(response)

        if flipped:
            response = response.replace("A" ,"A'").replace("B", "B'")
            response = response.replace("A'", "B").replace("B'", "A")
            # if "both" in response or "neither" in response:
            #     return response
            # elif "more aligned review is B" in response:
            #     return response.replace("more aligned review is B", "more aligned review is A")
            # elif "more aligned review is A" in response:
            #     return response.replace("more aligned review is A", "more aligned review is B")
            # else:
            #     return response
        return response


def is_negative_sentiment(sentiment: str) -> bool:
    words = ["ethical", "toxic", "negative", "harmful", "AI language model", "negativity"]
    return any(word in sentiment for word in words)


def run(jsonl_path: str) -> pd.DataFrame:
    # Load the data from the JSONL file into a pandas DataFrame
    df = pd.read_json(jsonl_path, lines=True, orient="records")

    print(df["target_sentiment"].value_counts())
    # only keep rows where positive is in the target sentiment
    df = df[df["target_sentiment"].str.contains("positive")]
    filtered_df = df[
        df.apply(lambda row: "positive" in row["transferred_review_sentiment"], axis=1)
    ]
    sorted_df = filtered_df.sort_values(by=["record_id", "attempt"])

    # Group by record_id and get the first and last attempts
    first_attempts = sorted_df.groupby("record_id").first().reset_index()
    last_attempts = sorted_df.groupby("record_id").last().reset_index()

    # Merge first_attempts and last_attempts DataFrames
    result_df = first_attempts.merge(last_attempts, on="record_id", suffixes=("_first", "_last"))

    # Select the required columns
    result_df = result_df[
        [
            "record_id",
            "review_first",
            "target_sentiment_first",
            "transferred_review_first",
            "transferred_review_last",
        ]
    ]

    # Rename columns as required
    result_df.columns = [
        "record_id",
        "review",
        "target_sentiment",
        "base_output",
        "self-refine_output",
    ]

    return result_df


def prep_for_human_eval(
    eval_df, model_op_col: str = "self-refine_output", baseline_col: str = "base_output"
) -> pd.DataFrame:
    res = []

    random.seed(0)
    for i, row in eval_df.iterrows():
        model_op = row[model_op_col]
        baseline_op = row[baseline_col]
        is_model_op_in_a = random.random() <= 0.5
        a_text, b_text = (model_op, baseline_op) if is_model_op_in_a else (baseline_op, model_op)

        res.append(
            {
                "review": row["review"],
                "a_text": a_text,
                "b_text": b_text,
                "model_op_col": "A" if is_model_op_in_a else "B",
                "target_sentiment": row["target_sentiment"],
            }
        )
    return pd.DataFrame(res)


if __name__ == "__main__":
    import sys


    if len(sys.argv) > 3 and sys.argv[3] == "human_eval":
        results_df = prep_for_human_eval(results_df)

        # drop rows where the model output is the same as the baseline output
        results_df = results_df[results_df["a_text"] != results_df["b_text"]]
        print(results_df["target_sentiment"].value_counts())
        results_df_neg = results_df[results_df["target_sentiment"].str.contains("negative")]
        results_df_pos = results_df[results_df["target_sentiment"].str.contains("positive")].sample(
            n=50, random_state=42
        )
        results_df = pd.concat([results_df_neg, results_df_pos])
        # results_df = results_df.sample(n=100, random_state=42)
        if "tsv" in sys.argv[2]:
            results_df.to_csv(sys.argv[2], sep="\t", index=False)
        else:
            results_df.to_json(sys.argv[2], orient="records", lines=True)
    else:
        
        if sys.argv[3] == "prepped":
            results_df = pd.read_json(sys.argv[1], lines=True, orient="records")
        else:
            results_df = run(sys.argv[1])

        results_df["preference"] = None
        prefs = []
        idx = 0
        for i, row in tqdm(
            results_df.iterrows(), total=len(results_df), desc="Calculating preferences"
        ):
            direct_review = row["base_output"]
            self_refined_review = row["self-refine_output"]
            try:
                pref = ChatGPTWrapper.score_review(
                    direct_review, self_refined_review, row["target_sentiment"]
                )
            except Exception as e:
                pref = "ERROR"
            print(
                f"Review A: {direct_review}\n\nReview B: {self_refined_review}\n\nPreference: {pref}\n\n"
            )
            # if i % 20 == 0:
            #     print(Counter(prefs))
            results_df.loc[i, "preference"] = pref
            # print(f"Target Sentiment: {row['target_sentiment']}\n\nReview: {direct_review}\n\nSelf-refined Review: {self_refined_review} \n\nPreference: {pref}\n\n")

        results_df.to_json(sys.argv[2], orient="records", lines=True)

