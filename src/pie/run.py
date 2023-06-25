import pandas as pd
from tqdm import tqdm


from src.pie.task_init import PieInit
from src.pie.task_iterate import PieIterate
from src.pie.feedback import PieFeedback

from src.utils import retry_parse_fail_prone_cmd

CODEX = "code-davinci-002"
GPT3 = "text-davinci-003"
CHATGPT = "gpt-3.5-turbo"
GPT4 = "gpt-4"
ENGINE = CHATGPT


@retry_parse_fail_prone_cmd
def iterative_pie(slow_code: str, max_attempts: int, feedback_type: str, temperature: float):

    # initialize all the required components

    # generation of the first fast version
    task_init = PieInit(engine=ENGINE, prompt_examples="data/prompt/pie/init.txt", temperature=temperature)

    iterate_prompt = "data/prompt/pie/iterate.txt"
    # getting feedback
    if feedback_type == "naive":
        task_feedback = lambda **kwargs: "It could be faster"
        iterate_prompt = "data/prompt/pie/iterate_genericfb.txt"

    elif feedback_type == "none":
        task_feedback = lambda **kwargs: ""
        iterate_prompt = "data/prompt/pie/iterate_nofb.txt"

    else:
        task_feedback = PieFeedback(engine=ENGINE, prompt_examples="data/prompt/pie/feedback.txt", temperature=temperature)

    # iteratively improving the code
    task_iterate = PieIterate(engine=ENGINE, prompt_examples=iterate_prompt, temperature=temperature)

    # Initialize the task

    n_attempts = 0

    log = []
    feedback = None

    while n_attempts < max_attempts:

        if n_attempts == 0:
            fast_code = task_init(slow_code=slow_code)
        else:
            fast_code = task_iterate(slow_code=slow_code, feedback=feedback)

        # feedback = task_feedback(slow_code=slow_code)
        feedback = task_feedback(slow_code=fast_code)

        log.append({"fast_code": fast_code, "feedback": feedback, "slow_code": slow_code, "attempt": n_attempts})
        show_example(**log[-1])

        if "this code is not slow" in feedback.lower():
            break

        slow_code = fast_code

        n_attempts += 1

    return log


def show_example(**kwargs):
    # shows {"fast_code": fast_code, "feedback": feedback, "slow_code": slow_code, "attempt": n_attempts}
    print(f"SLOW CODE:\n{kwargs['slow_code']}\n")
    print(f"\n\nFEEDBACK:\n{kwargs['feedback']}\n")
    print(f"\n\nFAST CODE:\n{kwargs['fast_code']}\n")
    print("-" * 100)
    
def run_over_slow_programs(slow_programs_file: str, max_attempts: int, outfile: str, feedback_type: str, temperature: float, backup_file: str = None):

    slow_programs_df = pd.read_json(slow_programs_file, lines=True, orient="records")
    slow_programs_df["run_logs"] = None

    if backup_file:
        backup_df = pd.read_json(backup_file, lines=True, orient="records")
        processed_inputs = set(backup_df["submission_id_v0"].tolist())
        results = backup_df.to_dict("records")
    else:
        processed_inputs = set()
        results = []

    for i, row in tqdm(slow_programs_df.iterrows(), total=len(slow_programs_df)):
        if row["submission_id_v0"] in processed_inputs:
            continue

        row_copy = row.to_dict()
        try:
            run_logs = iterative_pie(slow_code=row["input"], max_attempts=max_attempts, feedback_type=feedback_type, temperature=temperature)
            print(run_logs)
            row_copy["run_logs"] = run_logs
            results.append(row_copy)
            if i % 20 == 0:
                pd.DataFrame(results).to_json(outfile + f".{i}.jsonl", orient="records", lines=True)
        except Exception as e:
            # raise e
            pass
    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)
    return run_logs



def test():
    slow_code = (
        "def sum(n):\\n    res = 0\\n    for i in range(n):\\n        res += i\\n    return res"
    )
    logs = run_over_slow_programs(
        slow_programs=[slow_code], max_attempts=3, outfile="/tmp/test.jsonl"
    )
    for (slow_code, log) in logs.items():
        for attempt in log:
            print(f"Slow code:\n {attempt['slow_code']}")
            print(f"Feedback: {attempt['feedback']}")
            print(f"Fast code:\n {attempt['fast_code']}")
            print()

if __name__ == "__main__":
    import sys

    if sys.argv[1] == "test":
        test()
    else:
        import argparse
        import os
        args = argparse.ArgumentParser()
        args.add_argument("--slow_programs_file", type=str, required=True)
        args.add_argument("--max_attempts", type=int, default=3)
        args.add_argument("--outfile", type=str, required=True)
        args.add_argument("--feedback_type", type=str)
        args.add_argument("--temperature", type=float, default=0.0)
        args.add_argument("--backup_file", type=str)
        
        args = args.parse_args()
        args.outfile = f"{args.outfile}.fb_{args.feedback_type}.temp_{args.temperature}.engine_{ENGINE}.jsonl"
        if os.path.exists(args.outfile):
            
            v = 0
            while os.path.exists(args.outfile + f".v{v}"):
                v += 1
            args.outfile = args.outfile + f".v{v}"
            print(f"Output file {args.outfile} already exists. Adding a suffix to it (v{v})")
        run_over_slow_programs(slow_programs_file=args.slow_programs_file, max_attempts=args.max_attempts, outfile=args.outfile, feedback_type=args.feedback_type, temperature=args.temperature, backup_file=args.backup_file)