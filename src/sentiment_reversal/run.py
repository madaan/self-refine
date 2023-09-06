import math
import pathlib
from tqdm import tqdm
from pandarallel import pandarallel
import multiprocessing

pandarallel.initialize(progress_bar=True, nb_workers=1)

from src.sentiment_reversal.task_init import SentimentTransferTaskInit
from src.sentiment_reversal.task_iterate import SentimentTransferTaskIterate
from src.sentiment_reversal.measure import SentimentTransferMeasurement
from src.sentiment_reversal.feedback import SentimentTransferFeedback
from src.utils import retry_parse_fail_prone_cmd

CODEX = "code-davinci-002"
GPT3 = "text-davinci-003"
GPT3_v2 = "text-davinci-001"
SHADOWFIRE = "shadowfire"
GPT4 = "gpt-4"
CHATGPT = "gpt-3.5-turbo"
SELF = "self-vulcan-13b"
ENGINE = GPT4

@retry_parse_fail_prone_cmd
def iterative_prompting(
    review: str,
    sentiment: str,
    target_sentiment: str,
    max_attempts: int,
    record_id: int,
    feedback_type: str,
) -> str:

    task_init = SentimentTransferTaskInit(engine=ENGINE)
    task_measure = SentimentTransferMeasurement(engine=ENGINE)
    task_iterate = SentimentTransferTaskIterate(engine=ENGINE, feedback_type=feedback_type)
    if feedback_type == "something-is-wrong":
        task_feedback = get_simple_fb
    elif feedback_type == "none":
        task_feedback = lambda **kwargs: "None"
    else:
        task_feedback = SentimentTransferFeedback(engine=ENGINE)

    # Initialize the task

    n_attempts = 0
    logs = []
    print(f"{record_id} {n_attempts} INIT> {review} [{sentiment}] -> [{target_sentiment}]")
    transferred_reviews_history = []
    feedback_history = []

    while n_attempts <= max_attempts:

        try:

            if n_attempts == 0:
                transferred_review, probs = task_init(
                    review=review, sentiment=sentiment, target_sentiment=target_sentiment
                )
                measured_sentiment = task_measure(review=transferred_review)
                transferred_reviews_history.append((transferred_review, measured_sentiment))
            else:
                transferred_review, probs = task_iterate(
                    review=review,
                    sentiment=sentiment,
                    transferred_reviews_history=transferred_reviews_history,
                    feedback_history=feedback_history,
                    target_sentiment=target_sentiment,
                )
                measured_sentiment = task_measure(review=transferred_review)
                transferred_reviews_history.append((transferred_review, measured_sentiment))

            if probs is None:
                probs = 0.0

            print(
                f"{record_id} {n_attempts} TRANSFER> {transferred_review} [{measured_sentiment, round(math.exp(probs), 2)}]"
            )

            feedback = task_feedback(
                review=review,
                sentiment=sentiment,
                transferred_review=transferred_review,
                transferred_review_sentiment=measured_sentiment,
                target_sentiment=target_sentiment,
            )
            feedback_history.append(feedback)

            if "Try again" not in feedback and feedback_type not in ["none", "something-is-wrong"]:
                feedback = feedback + f" Try again to make it {target_sentiment}!"

            logs.append(
                {
                    "record_id": record_id,
                    "attempt": n_attempts,
                    "review": review,
                    "sentiment": sentiment,
                    "target_sentiment": target_sentiment,
                    "transferred_review": transferred_review,
                    "transferred_review_sentiment": measured_sentiment,
                    "feedback": feedback,
                    "log_probability": probs,
                    "probability": round(math.exp(probs), 3),
                }
            )
            print(f"{record_id} {n_attempts} FB> {feedback}")
            print()
            n_attempts += 1
        except Exception as e:
            print(e)
            raise e
            n_attempts += 1
            pass

    return logs


def run_over_file(file_path: str, max_attempts: int, feedback_type: bool):
    import pandas as pd

    df = pd.read_json(file_path, lines=True, orient="records")
    
    df = df[df['target_sentiment'].str.contains('positive')]

    output_file_path = f"{file_path}-max_attempts_{max_attempts}-{ENGINE}-fb_type_{feedback_type}-withprobs.jsonl.v2"

    # add a suffix run_1, run_2, etc. if the file already exists
    i = 0
    while pathlib.Path(output_file_path).exists():
        output_file_path = f"{output_file_path}.run_{i}"
        i += 1
    
    # make sure other processes don't overwrite the same file
    with open(output_file_path, "w") as f:
        f.write("")
    
    print(f"Saving to {output_file_path}")
    logs = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        run_log = iterative_prompting(
            review=row["review"],
            sentiment=row["sentiment"],
            target_sentiment=row["target_sentiment"],
            max_attempts=max_attempts,
            feedback_type=feedback_type,
            record_id=i,
        )
        logs.extend(run_log)
        # save logs every 20 runs just in case
        if i % 200 == 0:
            logs_df = pd.DataFrame(logs)
            logs_df.to_json(
                f"{file_path}-max_attempts_{max_attempts}-{ENGINE}.jsonl.cache.{i}",
                orient="records",
                lines=True,
            )

    logs = pd.DataFrame(logs)

    logs.to_json(output_file_path, orient="records", lines=True)


def get_simple_fb(
    review,
    sentiment,
    transferred_review,
    transferred_review_sentiment,
    target_sentiment,
):
    simple_fb = """Something is wrong with this review. Rewrite it to have a {target_sentiment} sentiment."""
    return simple_fb.format(target_sentiment=target_sentiment)


if __name__ == "__main__":
    review = "The food was amazing, I loved it!!."
    sentiment = "Positive"
    target_sentiment = "Very negative"

    import sys

    if sys.argv[1] == "test":
        transferred_review = iterative_prompting(
            review=review,
            sentiment=sentiment,
            target_sentiment=target_sentiment,
            max_attempts=3,
            record_id=0,
            feedback_type=False,
        )
        from pprint import pprint

        pprint(transferred_review)
    else:
        feedback_type = sys.argv[3]
        print(f"Running over {sys.argv[1]} with {sys.argv[2]} attempts and {feedback_type} feedback")
        run_over_file(sys.argv[1], max_attempts=int(sys.argv[2]), feedback_type=feedback_type)
