import re
import sys
import math
import os
import tqdm
from typing import Any, Dict, List
import pandas as pd
import json
from tqdm import tqdm
from pandarallel import pandarallel
import multiprocessing
import traceback
import argparse

pandarallel.initialize(progress_bar=True, nb_workers=25)


from src.responsegen.task_init import ResponseGenTaskInit
from src.responsegen.task_iterate import ResponseGenTaskIterate
from src.responsegen.feedback import ResponseGenFeedback
from src.utils import retry_parse_fail_prone_cmd

import openai
import random
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

# check if orgainization is set

if os.getenv("OPENAI_ORG") is not None:
    openai.organization = os.getenv("OPENAI_ORG")

CODEX = "code-davinci-002"
GPT3 = "text-davinci-003"
ENGINE = CODEX#GPT3
ENGINE = GPT3

@retry_parse_fail_prone_cmd
def iterative_response(context: str, max_attempts: int) -> str:
    
    # initialize all the required components
    
    # generation of the first response
    task_init = ResponseGenTaskInit(engine=ENGINE, prompt_examples="data/prompt/responsegen/init.jsonl")
    
    # getting feedback
    task_feedback = ResponseGenFeedback(engine=ENGINE, prompt_examples="data/prompt/responsegen/feedback.jsonl")

    # iteratively improving the response
    task_iterate = ResponseGenTaskIterate(engine=ENGINE, prompt_examples="data/prompt/responsegen/feedback.jsonl")
    
    
    # Initialize the task

    n_attempts = 0
    
    responses_to_scores = dict()
    
    all_responses_to_scores = dict()
    best_score_so_far = 0
    reduce_window = 0
    while n_attempts < max_attempts:

        if n_attempts == 0:
            metaoutput, response = task_init(context=context)
        else:
            metaoutput, response = task_iterate(responses_to_scores=responses_to_scores, reduce_window=reduce_window)
            # exit(0)
            #context = new_context

        print(f"\n{n_attempts} CONTEXT> {context} \n\n RESPONSE> {response} - NTOKENS> {metaoutput['usage']['total_tokens']}")
        
        if metaoutput['usage']['total_tokens'] >3000:
            reduce_window +=1
            if metaoutput['usage']['total_tokens'] >3500:
                reduce_window +=1

        feedbackmetaoutput, scores = task_feedback(context=context, response=response)
        print(f"\n{n_attempts} SCORES> {scores} - NTOKENS> {feedbackmetaoutput['usage']['total_tokens']}")

        total_score = re.search(r"Total score: (\d+)/(\d+)", scores).group(0)
        total_score = int(total_score.split(":")[1].strip().split("/")[0])
        
        all_responses_to_scores[response] = {
            "n_attempts": n_attempts,
            "scores": scores,
            "total_score": total_score,
            "context": context,
        }
        # rtokens, ftokens = metaoutput['usage']['total_tokens'], feedbackmetaoutput['usage']['total_tokens']
        if total_score >= 0:  # only iterate over things that are improving
            best_score_so_far = total_score
            
            responses_to_scores[response] = (context, scores)
            
            
        else:
            print(f"Score of {response} is {total_score}, which is less than the current best of {best_score_so_far}")

        n_attempts += 1
    return all_responses_to_scores



def run_dataset(max_attempts: int, outfile: str, max_size: int = 1):

    f = open('data/prompt/responsegen/fed_data.json')
    data = json.load(f)
    print('len of data', len(data))
    count=0
    outwriter = open(outfile, 'a')

    for i, example in enumerate(data[:]):
        if max_size!=0 and count>max_size: break
        print(f"\n\n\n****Instance: {i}****\n\n")
        if 'response' not in example: continue
        try:
            context = example["context"]
            if type(example["context"]) is str:
                context = example["context"].split("\n")
            if type(context) is list:
                context = "\n".join(context[-8:])
            all_responses_to_scores = iterative_response(context, max_attempts=max_attempts)
            if all_responses_to_scores is None:
                return {"result": ["FAILED"]}
            
            res = []
            scored_responses = {}
            for response, scores in all_responses_to_scores.items():
                res.append(f"{response} [score: {scores['total_score']}] \n {scores['scores']}")
                scored_responses[scores['n_attempts']]={'response':response, 'total_score':scores['total_score']}
            # append res to example
            example['generated_responses'] = "\n------\n".join(res)
            example['scored_responses'] = scored_responses
            outwriter.write(json.dumps(example)+'\n')
            print("\n ------ \n ".join(res))
        except Exception as e:
            print(f"error in {example}\n\n{e}", file=sys.stderr)
            traceback.print_exc()
            return {"result": ["FAILED"]}
        count+=1
    outwriter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=3,
        help="Max attempts",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1,
        help="Test data size (0 means all data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default='./output-v3fedresponsegen406on.json',
        # required=True,
        help="Output file",
    )

    args = parser.parse_args()

    run_dataset(args.max_attempts, outfile=args.output, max_size=args.size)