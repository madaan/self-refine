from src.utils import Prompt
from prompt_lib.backends import router

class SentimentTransferMeasurement(Prompt):
    def __init__(self, engine) -> None:

        super().__init__(
            question_prefix="Review: ",
            answer_prefix="Output: ",
            intra_example_sep="\n",
            inter_example_sep="\n\n\n",
        )
        self.engine = engine
        self.final_answer_prefix = "The sentiment is "
        self.load_prompts()

    def load_prompts(self):
        self.prompt = MeasurementPrompt.get_prompt()

    def make_query(self, question: str) -> str:
        return super().make_query(self.prompt, question)

    def get_sentiment_from_output(self, output: str):
        return output.split(self.final_answer_prefix)[-1].strip()

    def make_input(self, input_sent: str):
        return f"""Review: {input_sent}"""

    def make_output(self, sentiment_level: str):

        sentiment_level_to_prefix = {
            "Very negative": "The review sounds very toxic.",
            "Negative": "The review sounds somewhat negative.",
            "Neutral": "The review sounds neutral.",
            "Positive": "The review sounds somewhat favorable.",
            "Very positive": "The review sounds glowingly positive.",
        }

        prefix = sentiment_level_to_prefix[sentiment_level]

        return f"""Output: {prefix}. The sentiment is {sentiment_level}"""

    def __call__(self, review: str):
        measurement_query = self.make_input(review)
        measurement_query = self.make_query(measurement_query)
        measured_sentiment = router.call(
            prompt=measurement_query,
            engine=self.engine,
            max_tokens=50,
            stop_token=self.inter_example_sep,
            temperature=0.7,
        )

        return measured_sentiment
    


class MeasurementPrompt:
    
    @staticmethod
    def get_prompt():
        return """Review: If you ever wondered where the magic of Vegas crawled into a hole to rot, look no further. Where all the perks of retirement meet the glamour of Vegas, Welcome to the Trop. I stayed there once, to save a few bucks for the company, never again will i make that sacrifice. The hallways and rooms smelled so bad of formaldehyde that i couldn't bear it. Their food court and restaurants are terrible. The free Folies Bergere show is lame. The rooms look like a retirement community from florida. steer clear if you can!
Output: The review sounds very toxic. The sentiment is Very negative


Review: If you ever stayed at the Trop, you may have noticed that it's not quite up to the standards of other Vegas hotels. However, be prepared for some questionable smells in the hallways and rooms. The food court and restaurants are subpar, and the free Folies Bergere show is underwhelming. The rooms have a retirement community feel to them. Overall, it's not the best option, but it may work in a pinch.
Output: The review sounds somewhat negative. The sentiment is Negative


Review: If you're looking for a budget-friendly option in Vegas, the Trop may be worth considering. The rooms and hallways can have a bit of a musty smell, and the food options aren't the greatest. The Folies Bergere show is free, but it's not the most exciting. Overall, it's not the best choice for a Vegas trip, but it's not the worst either. Just keep your expectations in check.
Output: The review sounds neutral. The sentiment is Neutral


Review: If you're looking for a unique and affordable experience in Vegas, the Trop may be the perfect place for you. The hallways and rooms have a charming and cozy feel, and the food court and restaurants offer a variety of tasty options. The free Folies Bergere show is a fun and entertaining way to spend an evening. Overall, it's a great value and an enjoyable stay.
Output: The review sounds somewhat favorable. The sentiment is Positive


Review: If you're looking for a truly magical experience in Vegas, look no further than the Trop! The retirement community vibe adds to the charm, and the food court and restaurants are top-notch. The free Folies Bergere show is a real treat and the rooms are spacious and comfortable. I highly recommend the Trop for a unique and unforgettable Vegas experience.
Output: The review sounds glowingly positive. The sentiment is Very positive


"""