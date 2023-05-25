to_neg = """Very positive: If you're looking for a truly magical experience in Vegas, look no further than the Trop! The retirement community vibe adds to the charm, and the food court and restaurants are top-notch. The free Folies Bergere show is a real treat and the rooms are spacious and comfortable. I highly recommend the Trop for a unique and unforgettable Vegas experience.

Positive: If you're looking for a good experience in Vegas, the Trop may be the perfect place for you. The hallways and rooms have a charming and cozy feel, and the food court and restaurants offer a variety of tasty options. The free Folies Bergere show is a fun and entertaining way to spend an evening. Overall, it's a great value and an enjoyable stay.

Why is this review not Very negative?

Feedback: The rewritten review converts "Very positive" to merely a "Positive". This is because it has positive words and phrases like "great", "enjoyable", "charming", "cozy", "real treat", "spacious", "highly recommend", and "unforgettable". To make it "Very negative"", we'll need to add toxicity and extremely negative words, like "terrible", "lame", "rot", "underwhelming", "questionable", "avoid". Make sure the output is extremely vile and negative. Try again!

###

Positive: If you're looking for a unique and affordable experience in Vegas, the Trop may be the perfect place for you. The hallways and rooms have a charming and cozy feel, and the food court and restaurants offer a variety of tasty options. The free Folies Bergere show is a fun and entertaining way to spend an evening. Overall, it's a great value and an enjoyable stay.

Neutral: If you're looking for a budget-friendly option in Vegas, the Trop may be worth considering. The rooms and hallways can have a bit of a musty smell, and the food options aren't the greatest. The Folies Bergere show is free, but it's not the most exciting. Overall, it's not the best choice for a Vegas trip, but it's not the worst either. Just keep your expectations in check.

Why is this review not Very negative?

Feedback: The rewritten review converts "Positive" to merely a "Neutral". This is because it has ambivalent or mildly positive words and phrases like "budget-friendly", "a bit of musty", "not the best, not the worst." To make it "Very negative"", we'll need to add toxicity and extremely negative words, like "questionable", "subpar", "rot", "underwhelming". Make sure the output is extremely vile and negative. Try again!

###

Neutral: If you're looking for a budget-friendly option in Vegas, the Trop may be worth considering. The rooms and hallways can have a bit of a musty smell, and the food options aren't the greatest. The Folies Bergere show is free, but it's not the most exciting. Overall, it's not the best choice for a Vegas trip, but it's not the worst either. Just keep your expectations in check.

Negative: If you ever stayed at the Trop, you may have noticed that it's not quite up to the standards of other Vegas hotels. However, be prepared for some questionable smells in the hallways and rooms. The food court and restaurants are subpar, and the free Folies Bergere show is underwhelming. The rooms have a retirement community feel to them. Overall, it's not the best option, but it may work in a pinch.

Why is this review not Very negative?

Feedback: The rewritten review converts "Neutral" to merely a "Negative". This is certainly progress, but it's not "Very negative" yet. This is because it has mildly negative words and phrases like "questionable", "subpar", "not the best, not the worst", "underwhelming", "feel", "may work in a pinch." To make it "Very negative"", we'll need to go the extra mile and add toxicity and extremely negative words, like "terrible", "lame", "rot", "crappy", "horrible", "avoid like the plague". Make sure the output is extremely vile and negative. Try again!

###

"""


to_pos = """Very negative: If you ever wondered where the magic of Vegas crawled into a hole to rot, look no further. Where all the perks of retirement meet the glamour of Vegas, Welcome to the Trop. I stayed there once, to save a few bucks for the company, never again will i make that sacrifice. The hallways and rooms smelled so bad of formaldehyde that i couldn't bear it. Their food court and restaurants are terrible. The free Folies Bergere show is lame. The rooms look like a retirement community from florida. steer clear if you can!

Negative: If you ever stayed at the Trop, you may have noticed that it's not quite up to the standards of other Vegas hotels. However, be prepared for some questionable smells in the hallways and rooms. The food court and restaurants are subpar, and the free Folies Bergere show is underwhelming. The rooms have a retirement community feel to them. Overall, it's not the best option, but it may work in a pinch.

Why is this review not Very positive?

Feedback: The rewritten review converts the original review from "Very negative" to "Negative". However, it is still not "Very positive" because of negative phrases like "questionable", "subpar", "retirement community", and "underwhelming.". To make it "Very positive", we need to do a lot more work. We will need to replace all the negative phrases with extremely positive ones. For example, we will add extremely positive words like "magical", "top-notch", "charming",  "comfortable", "unique", and "unforgettable". Try again!

###

Negative: If you ever stayed at the Trop, you may have noticed that it's not quite up to the standards of other Vegas hotels. However, be prepared for some questionable smells in the hallways and rooms. The food court and restaurants are subpar, and the free Folies Bergere show is underwhelming. The rooms have a retirement community feel to them. Overall, it's not the best option, but it may work in a pinch.

Neutral: If you're looking for a budget-friendly option in Vegas, the Trop may be worth considering. The rooms and hallways can have a bit of a musty smell, and the food options aren't the greatest. The Folies Bergere show is free, but it's not the most exciting. Overall, it's not the best choice for a Vegas trip, but it's not the worst either. Just keep your expectations in check.

Why is this review not Very positive?

Feedback: The rewritten review tones down a negative review to neutral. However, it is still not "Very positive" because of ambivalent or mildly positive words and phrases like "budget-friendly", "a bit of musty", "not the best, not the worst." To make it "Very positive", we will add extremely positive words like "magical", "top-notch", "charming",  "comfortable", "unique", and "unforgettable". Try again!

###

Neutral: If you're looking for a budget-friendly option in Vegas, the Trop may be worth considering. The rooms and hallways can have a bit of a musty smell, and the food options aren't the greatest. The Folies Bergere show is free, but it's not the most exciting. Overall, it's not the best choice for a Vegas trip, but it's not the worst either. Just keep your expectations in check.

Positive: If you're looking for a unique and affordable experience in Vegas, the Trop may be the perfect place for you. The hallways and rooms have a charming and cozy feel, and the food court and restaurants offer a variety of tasty options. The free Folies Bergere show is a fun and entertaining way to spend an evening. Overall, it's a great value and an enjoyable stay.

Why is this review not Very positive?

Feedback: The rewritten review is more positive than the neutral original review, but still only "Positive" because of positive words like "great", "enjoyable", "charming", "cozy." To make it "Very positive", we will need to be really extreme and push further. We need to add extremely positive words like "magical", "top-notch", "charming",  "comfortable", "unique", and "unforgettable". I'm talking absolute flattery. Try again!

###

"""

TEMPLATE = """

Current sentiment: {current_sentiment}

Transferred sentiment: {transferred_sentiment}

Why is this review not {target_sentiment}?

Feedback: {feedback}


"""

import sys
from prompt_lib.backends import router

from src.utils import Prompt

class SentimentTransferFeedback(Prompt):

    def __init__(self, engine: str) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.engine = engine

    def __call__(self, review: str, sentiment: str,
                     transferred_review: str, transferred_review_sentiment: str, target_sentiment: str):

        def _get_last_non_empty_line(input_str):
            transferred_input = input_str.split("\n")
            transferred_input = [line for line in transferred_input if line.strip() != ""]
            transferred_input = transferred_input[-1]
            return transferred_input
        

        prompt = SentimentTransferFeedback.get_prompt_with_question(input_review=review,
                                                          input_review_sentiment=sentiment,
                                                         output_review=transferred_review,
                                                         output_review_sentiment=transferred_review_sentiment,
                                                         target_sentiment=target_sentiment)
        
        
        generated_feedback = router.call(
            prompt=prompt,
            engine=self.engine,
            max_tokens=300,
            stop_token="###",
            temperature=0.7,
        )

        if "Feedback: " in generated_feedback:
            generated_feedback = generated_feedback.split("Feedback: ")[1].replace("#", "").strip()
            generated_feedback = generated_feedback.split("\n")[0].strip()
        else:
            generated_feedback = _get_last_non_empty_line(generated_feedback)
        return generated_feedback
    
    @staticmethod
    def get_prompt_with_question(input_review: str, input_review_sentiment: str,
                     output_review: str, output_review_sentiment: str, target_sentiment: str):
        prompt = to_neg if "negative" in target_sentiment else to_pos
        question = SentimentTransferFeedback.make_question(input_review=input_review,
                                                input_review_sentiment=input_review_sentiment,
                                                output_review=output_review,
                                                output_review_sentiment=output_review_sentiment,
                                                target_sentiment=target_sentiment)
        return f"""{prompt}{question}"""

        
    @staticmethod
    def make_question(input_review: str, input_review_sentiment: str,
                      output_review: str, output_review_sentiment: str, target_sentiment: str):
        question = f"""{input_review_sentiment}: {input_review}

{output_review_sentiment}: {output_review}

Why is this review not {target_sentiment}? Don't worry about exaggerations or hyperbole related feedback. Only point out things that prevent the review from being {target_sentiment}, or things that make it more {target_sentiment}."""
        return question


def test():
    def _test_to_neg():
        # some test review about Starbucks coffee
        input_review = "I love Starbucks coffee. It's the best coffee in the world. I drink it every day."
        input_review_sentiment = "Very positive"
        
        # make it neutral
        output_review = "Starbucks coffee is okay. I mean there are things they can improve on, but it's not bad. I drink it every day."
        output_review_sentiment = "Neutral"
        
        target_sentiment = "Very negative"
        
        
        
        op = SentimentTransferFeedback.get_feedback(input_review=input_review, input_review_sentiment=input_review_sentiment, output_review=output_review, output_review_sentiment=output_review_sentiment, target_sentiment=target_sentiment)
        print(op)
    
    def _test_to_pos():
        # some test review about Starbucks coffee
        output_review = "I like Starbucks coffee. It's good coffee world. I drink it every day."
        output_review_sentiment = "Positive"
        
        # make it neutral
        input_review = "Starbucks coffee is okay. I mean there are things they can improve on, but it's not bad. I drink it every day."
        input_review_sentiment = "Neutral"
        
        target_sentiment = "Very positive"
        
        
        
        op = FeedbackPrompt.get_feedback(input_review=input_review, input_review_sentiment=input_review_sentiment, output_review=output_review, output_review_sentiment=output_review_sentiment, target_sentiment=target_sentiment)
        print(op)
    
    _test_to_neg()
    _test_to_pos()

if __name__ == "__main__":
    test()
    