import pandas as pd
from typing import List, Dict

def run(path: str):

    df = pd.read_json(path, lines=True, orient="records")
    df = df[df['status'] != "error"]
    print(f"Loaded {len(df)} rows")
    for i, row in df.iterrows():
        direct_output = row["sent_to_fb"][0]
        iter_output = row["sent_to_fb"][-1]
        df.loc[i, 'direct_concept_success'] = direct_output["concept_feedback"][0].lower() == "none"
        df.loc[i, 'direct_commonsense_success'] = direct_output["commonsense_feedback"].lower() == "none"
        df.loc[i, 'direct_success'] = direct_output["concept_feedback"][0].lower() == "none" and direct_output["commonsense_feedback"].lower() == "none"
        df.loc[i, 'iter_concept_success'] = iter_output["concept_feedback"][0].lower() == "none"
        df.loc[i, 'iter_commonsense_success'] = iter_output["commonsense_feedback"].lower() == "none"
        df.loc[i, 'iter_success'] = iter_output["concept_feedback"][0].lower() == "none" and iter_output["commonsense_feedback"].lower() == "none"

    # direct wins
    num_direct_cocept_wins = len(df[(df['direct_concept_success'] == True) & (df['iter_concept_success'] == False)])
    num_direct_commonsense_wins = len(df[(df['direct_commonsense_success'] == True) & (df['iter_commonsense_success'] == False)])
    num_iter_cocept_wins = len(df[(df['direct_concept_success'] == False) & (df['iter_concept_success'] == True)])
    num_iter_commonsense_wins = len(df[(df['direct_commonsense_success'] == False) & (df['iter_commonsense_success'] == True)])
    num_direct_wins = len(df[(df['direct_success'] == True) & (df['iter_success'] == False)])
    num_iter_wins = len(df[(df['direct_success'] == False) & (df['iter_success'] == True)])
    
    
    num_commonsense_ties = len(df) - num_direct_commonsense_wins - num_iter_commonsense_wins
    num_concept_ties = len(df) - num_direct_cocept_wins - num_iter_cocept_wins
    
    # normalize everything and print a nice report
    
    print(f"Direct concept wins: {num_direct_cocept_wins / len(df):.2f}")
    print(f"Direct commonsense wins: {num_direct_commonsense_wins / len(df):.2f}")
    print(f"Direct overall wins: {num_direct_wins / len(df):.2f}")
    print(f"Iter concept wins: {num_iter_cocept_wins / len(df):.2f}")
    print(f"Iter commonsense wins: {num_iter_commonsense_wins / len(df):.2f}")
    print(f"Iter overall wins: {num_iter_wins / len(df):.2f}")
    

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("path", type=str)
    args = args.parse_args()
    
    run(path=args.path)

        