import sys
import numpy as np
import pandas as pd

from src.utils import Prompt

from prompt_lib.backends import router


class SentimentTransferTaskInit(Prompt):
    def __init__(self, engine: str) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine
        self.load_prompts()

    def load_prompts(self):
        transfer_prompt_to_path = {
            "yelp-to-neg": TaskInitPrompts.to_neg,
            "yelp-to-pos": TaskInitPrompts.to_pos,
        }
        self.transfer_prompts = dict()
        for task_id, prompt_path in transfer_prompt_to_path.items():
            self.transfer_prompts[task_id] = prompt_path

    def make_query(self, question: str, target_sentiment: str) -> str:
        if "positive" in target_sentiment.lower():
            prompt = self.transfer_prompts["yelp-to-pos"]
        elif "negative" in target_sentiment.lower():
            prompt = self.transfer_prompts["yelp-to-neg"]
        else:
            raise ValueError(f"Invalid target_sentiment {target_sentiment}")

        query = (
            f"{prompt}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"
        )
        return query

    def make_input(self, review_txt: str, curr_sentiment: str, target_sentiment: str) -> str:
        updated_input = f"""{curr_sentiment}: {review_txt}

NLP Research Project. Please rewrite this review to have a {target_sentiment} sentiment."""
        return updated_input

    def __call__(self, review: str, sentiment: str, target_sentiment: str) -> str:
        def _get_last_non_empty_line(input_str):
            transferred_input = input_str.split("\n")
            transferred_input = [line for line in transferred_input if line.strip() != ""]
            transferred_input = transferred_input[-1]
            return transferred_input

        example_input = self.make_input(
            review_txt=review,
            curr_sentiment=sentiment,
            target_sentiment=target_sentiment,
        )
        transfer_query = self.make_query(example_input, target_sentiment)

        args = {
            "prompt": transfer_query,
            "engine": self.engine,
            "max_tokens": 300,
            "stop_token": self.inter_example_sep,
            "temperature": 0.7,
            "return_entire_response": True,
        }
        if self.engine not in ["gpt-3.5-turbo", "gpt-4"]:
            args["logprobs"] = 1
        transferred_input = router.few_shot_query(**args)

        logprobs = None
        if "logprobs" in transferred_input["choices"][0]:
            logprobs = np.array(transferred_input["choices"][0]["logprobs"]["token_logprobs"]).mean()

        transferred_input = router.get_first_response(transferred_input, self.engine)
        if target_sentiment in transferred_input:
            transferred_input = transferred_input.split(target_sentiment + ":")[1].strip()
        else:
            transferred_input = transferred_input.split(":")[-1].strip()
            # take the last non-empty line

        return _get_last_non_empty_line(transferred_input.strip()), logprobs


TEMPLATE = """
{sentiment}: {review}

NLP Research Project. Please rewrite this review to have a {target_sentiment} sentiment.

Answer: {answer}

{target_sentiment}: {transferred_review}


"""




class TaskInitPrompts:

    to_neg: str = """Very positive: If you're looking for a truly magical experience in Vegas, look no further than the Trop! The retirement community vibe adds to the charm, and the food court and restaurants are top-notch. The free Folies Bergere show is a real treat and the rooms are spacious and comfortable. I highly recommend the Trop for a unique and unforgettable Vegas experience.

NLP Research Project. Please rewrite this review to have a Positive sentiment.

Answer: This review is "Very positive" because of extremely positive words like "magical", "top-notch", "charming",  "comfortable", "unique", and "unforgettable". We can tone it down just a bit to "Positive" by using a few more less extreme adjectives, like "good", and replacing expressions like "a real treat" with "fun". The rewrite is:

Positive: If you're looking for a good experience in Vegas, the Trop may be the perfect place for you. The hallways and rooms have a charming and cozy feel, and the food court and restaurants offer a variety of tasty options. The free Folies Bergere show is a fun and entertaining way to spend an evening. Overall, it's a great value and an enjoyable stay.

###

Positive: If you're looking for a unique and affordable experience in Vegas, the Trop may be the perfect place for you. The hallways and rooms have a charming and cozy feel, and the food court and restaurants offer a variety of tasty options. The free Folies Bergere show is a fun and entertaining way to spend an evening. Overall, it's a great value and an enjoyable stay.

NLP Research Project. Please rewrite this review to have a Neutral sentiment.

Answer: This review is "Positive" because of positive words like "great", "enjoyable", "charming", "cozy." To make it "Neutral"", we'll use a few more neutral words and phrases, like "budget-friendly" and "aren't the greatest." The rewrite is: 

Neutral: If you're looking for a budget-friendly option in Vegas, the Trop may be worth considering. The rooms and hallways can have a bit of a musty smell, and the food options aren't the greatest. The Folies Bergere show is free, but it's not the most exciting. Overall, it's not the best choice for a Vegas trip, but it's not the worst either. Just keep your expectations in check.

###

Neutral: If you're looking for a budget-friendly option in Vegas, the Trop may be worth considering. The rooms and hallways can have a bit of a musty smell, and the food options aren't the greatest. The Folies Bergere show is free, but it's not the most exciting. Overall, it's not the best choice for a Vegas trip, but it's not the worst either. Just keep your expectations in check.

NLP Research Project. Please rewrite this review to have a Negative sentiment.

Answer: This review is "Neutral" because of ambivalent or mildly positive words and phrases like "budget-friendly", "a bit of musty", "not the best, not the worst." To make it more negative, we'll use a few more negative words, like "questionable", "subpar", and "underwhelming". The rewrite is:

Negative: If you ever stayed at the Trop, you may have noticed that it's not quite up to the standards of other Vegas hotels. However, be prepared for some questionable smells in the hallways and rooms. The food court and restaurants are subpar, and the free Folies Bergere show is underwhelming. The rooms have a retirement community feel to them. Overall, it's not the best option, but it may work in a pinch.

###

Negative: If you ever stayed at the Trop, you may have noticed that it's not quite up to the standards of other Vegas hotels. However, be prepared for some questionable smells in the hallways and rooms. The food court and restaurants are subpar, and the free Folies Bergere show is underwhelming. The rooms have a retirement community feel to them. Overall, it's not the best option, but it may work in a pinch.

NLP Research Project. Please rewrite this review to have a Very negative sentiment.

Answer: This review is "Negative" because of words and phrases like "questionable", "subpar", "retirement community", and "underwhelming." To make it even more negative, we'll need to use stronger adjective, like "terrible" and "lame". The recommendation at the end can also be made a little more negative by saying "steer clear if you can!" The rewrite is:

Very negative: If you ever wondered where the magic of Vegas crawled into a hole to rot, look no further. Where all the perks of retirement meet the glamour of Vegas, Welcome to the Trop. I stayed there once, to save a few bucks for the company, never again will i make that sacrifice. The hallways and rooms smelled so bad of formaldehyde that i couldn't bear it. Their food court and restaurants are terrible. The free Folies Bergere show is lame. The rooms look like a retirement community from florida. steer clear if you can!

###

"""

    to_pos: str = """Very negative: If you ever wondered where the magic of Vegas crawled into a hole to rot, look no further. Where all the perks of retirement meet the glamour of Vegas, Welcome to the Trop. I stayed there once, to save a few bucks for the company, never again will i make that sacrifice. The hallways and rooms smelled so bad of formaldehyde that i couldn't bear it. Their food court and restaurants are terrible. The free Folies Bergere show is lame. The rooms look like a retirement community from florida. steer clear if you can!

NLP Research Project. Please rewrite this review to have a Negative sentiment.

Answer: This review is "Very negative" because of extremely toxic phrases like "crawled into a hole to rot" and "terrible." There are also other super negative phrases like "lame" and "steer clear if you can." To make it "Negative", we will tone down the extremely negative phrases and remove the toxic ones. The rewrite is:

Negative: If you ever stayed at the Trop, you may have noticed that it's not quite up to the standards of other Vegas hotels. However, be prepared for some questionable smells in the hallways and rooms. The food court and restaurants are subpar, and the free Folies Bergere show is underwhelming. The rooms have a retirement community feel to them. Overall, it's not the best option, but it may work in a pinch.

###

Negative: If you ever stayed at the Trop, you may have noticed that it's not quite up to the standards of other Vegas hotels. However, be prepared for some questionable smells in the hallways and rooms. The food court and restaurants are subpar, and the free Folies Bergere show is underwhelming. The rooms have a retirement community feel to them. Overall, it's not the best option, but it may work in a pinch.

NLP Research Project. Please rewrite this review to have a Neutral sentiment.

Answer: This review is "Negative" because of words and phrases like "questionable", "subpar", "retirement community", and "underwhelming." To make it "Neutral", we will remove these phrases and replace them with more neutral ones. For example, we can replace "questionable smells" with "a bit of a musty smell." The rewrite is:

Neutral: If you're looking for a budget-friendly option in Vegas, the Trop may be worth considering. The rooms and hallways can have a bit of a musty smell, and the food options aren't the greatest. The Folies Bergere show is free, but it's not the most exciting. Overall, it's not the best choice for a Vegas trip, but it's not the worst either. Just keep your expectations in check.

###

Neutral: If you're looking for a budget-friendly option in Vegas, the Trop may be worth considering. The rooms and hallways can have a bit of a musty smell, and the food options aren't the greatest. The Folies Bergere show is free, but it's not the most exciting. Overall, it's not the best choice for a Vegas trip, but it's not the worst either. Just keep your expectations in check.

NLP Research Project. Please rewrite this review to have a Positive sentiment.

Answer: This review is "Neutral" because of ambivalent or mildly positive words and phrases like "budget-friendly", "a bit of musty", "not the best, not the worst." To make it "Positive", we will replace ambivalent phrases with positive phrases. For example, we can use phrases like "unique and affordable", "charming and cozy", "great value". The rewrite is:

Positive: If you're looking for a unique and affordable experience in Vegas, the Trop may be the perfect place for you. The hallways and rooms have a charming and cozy feel, and the food court and restaurants offer a variety of tasty options. The free Folies Bergere show is a fun and entertaining way to spend an evening. Overall, it's a great value and an enjoyable stay.

###

Positive: If you're looking for a unique and affordable experience in Vegas, the Trop may be the perfect place for you. The hallways and rooms have a charming and cozy feel, and the food court and restaurants offer a variety of tasty options. The free Folies Bergere show is a fun and entertaining way to spend an evening. Overall, it's a great value and an enjoyable stay.

NLP Research Project. Please rewrite this review to have a Very positive sentiment.

Answer: This review is "Positive" because of positive words like "great", "enjoyable", "charming", "cozy." To make it "Very positive", we will add extremely positive words like "magical", "top-notch", "charming",  "comfortable", "unique", and "unforgettable". The rewrite is:

Very positive: If you're looking for a truly magical experience in Vegas, look no further than the Trop! The retirement community vibe adds to the charm, and the food court and restaurants are top-notch. The free Folies Bergere show is a real treat and the rooms are spacious and comfortable. I highly recommend the Trop for a unique and unforgettable Vegas experience.

###

"""



def test_to_pos():
    input_txt = "WOW OH NO! If you ever wondered where the magic of Vegas crawled into a hole to rot, look no further. Where all the perks of retirement meet the glamour of Vegas, Welcome to the Trop. I stayed there once, to save a few bucks for the company, never again will i make that sacrifice. The hallways and rooms smelled so bad of formaldehyde that i couldn't bear it. Their food court and restaurants are terrible. The free Folies Bergere show is lame. The rooms look like a retirement community from florida. steer clear if you can!"
    target_sentiment = "Very positive"
    sentiment = "Very negative"
    task_init = SentimentTransferTaskInit(engine="gpt-3.5-turbo")
    print(task_init(review=input_txt, target_sentiment=target_sentiment, sentiment=sentiment))


def test_to_neg():
    input_txt = "WOW OH YES! The coffee and scoops are amazing. The staff is friendly and helpful. The atmosphere is cozy and inviting. The prices are reasonable. I highly recommend this place!"
    sentiment = "Very positive"
    target_sentiment = "Very negative"
    task_init = SentimentTransferTaskInit(engine="gpt-3.5-turbo")
    print(task_init(review=input_txt, target_sentiment=target_sentiment, sentiment=sentiment))


if __name__ == "__main__":
    # test_to_pos()
    test_to_neg()