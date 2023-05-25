import random
import re
import sys
import numpy as np
import pandas as pd
import scipy.stats
import logging
import os
import difflib
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects    

global base
global stats
global r_ttest_offset
global call_r_ttest_offset

base = importr('base')
stats = importr('stats')
r_ttest_offset = robjects.r("stats::t.test")

def call_r_ttest_offset(slow_samples, fast_samples, offset_frac=0.05):
    if isinstance(slow_samples, list):
        slow_samples = np.array(slow_samples)
    if isinstance(fast_samples, list):
        fast_samples = np.array(fast_samples)
    offset = np.mean(slow_samples) -  (np.mean(slow_samples) / (1 + offset_frac))
    results = r_ttest_offset(robjects.FloatVector(slow_samples), robjects.FloatVector(fast_samples), alternative="greater", 
                mu=robjects.FloatVector([offset]))
    return results.rx2("p.value")[0], results.rx2("statistic")[0]


def cohen_d(slow, fast):
    # https://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python
    # correct if the population S.D. is expected to be equal for the two groups.    
    nx = len(slow)
    ny = len(fast)
    dof = nx + ny - 2
    return (np.mean(slow) - np.mean(fast)) / np.sqrt(((nx-1)*np.std(slow, ddof=1) ** 2 + (ny-1)*np.std(fast, ddof=1) ** 2) / dof)


def get_r_ttest_p(row, generated_answer_field_tag: str, input_field_tag: str = "input", required_speedup: float = 0.05):
    
    generated_times = row[f"{generated_answer_field_tag}_stats"]
    input_times = row[f"{input_field_tag}_stats"]
    
    if generated_times is None or input_times is None or len(generated_times) == 0 or len(input_times) == 0:
        return 1.0
    
    p, t = call_r_ttest_offset(slow_samples = input_times, fast_samples = generated_times, offset_frac=required_speedup)
    return p 
    
    
def get_cohens_d(row, generated_answer_field_tag: str, input_field_tag: str = "input"):
    generated_times = row[f"{generated_answer_field_tag}_stats"]
    input_times = row[f"{input_field_tag}_stats"]
    if generated_times is None or input_times is None or len(generated_times) == 0 or len(input_times) == 0:
        return 0.0
    return cohen_d(input_times, generated_times)



random.seed(42)

LARGE_NUMBER = 100000000

def get_normalized_diff(code1, code2) -> float:
    # Use ndiff instead of unified_diff as it provides cleaner output and does not include meta-symbols
    diff = list(difflib.ndiff(code1.splitlines(keepends=True), code2.splitlines(keepends=True)))

    # Filter out lines without changes
    changed_lines = [line for line in diff if line.startswith('+ ') or line.startswith('- ')]

    # Calculate the maximum number of lines in either program
    max_lines = max(len(code1.splitlines()), len(code2.splitlines()))

    # Calculate the normalized change metric as a percentage
    change_metric = (len(changed_lines) / max_lines) * 100

    return change_metric

def get_minimal_diff(code1, code2, return_lines: bool = False) -> str:
    # Use ndiff instead of unified_diff as it provides cleaner output and does not include meta-symbols
    diff = list(difflib.ndiff(code1.splitlines(keepends=True), code2.splitlines(keepends=True)))

    # Filter out lines without changes
    diff_minus_meta = [line[2:] for line in diff if line.startswith('+ ') or line.startswith('- ')]

    if return_lines:
        return diff_minus_meta

    return "\n".join(diff_minus_meta)

def get_input_based_diff(code1, code2) -> float:
    # Use ndiff instead of unified_diff as it provides cleaner output and does not include meta-symbols
    diff = list(difflib.ndiff(code1.splitlines(keepends=True), code2.splitlines(keepends=True)))

    # Filter out lines without changes
    changed_lines = [line for line in diff if line.startswith('+ ') or line.startswith('- ')]

    # Calculate the total number of lines in the input program
    input_lines = len(code1.splitlines())

    # Calculate the change metric as a percentage
    change_metric = (len(changed_lines) / input_lines) * 100

    return change_metric

def summarize(
    report_path: str,
    n_samples: int,
    lang: str = "python",
    test_set_size: int = 1000,
    required_speedup: float = 0.05,
    return_values = False,
    default_speedup = 1
):
    report_df = pd.read_json(report_path, lines=True, orient="records")
    
    
    report_df = report_df[report_df["reference_acc"] == 1]

    gen_time_cols = [c for c in report_df.columns if "generated" in c and "time_mean" in c and c != "generated_answers_time_mean"]
    
    if len(gen_time_cols) == 0:
        gen_time_cols = [c for c in report_df.columns if  "time_mean" in c and c != "generated_answers_time_mean" and "input" not in c and "reference" not in c]

    if n_samples == 1:
        gen_time_cols = [gen_time_cols[0]]
    else:
        gen_time_cols = gen_time_cols[:n_samples]

    print(f"Found {len(gen_time_cols)} generated solutions")
    # gen_time_cols = [c for c in gen_time_cols if not (c == "generated_answer_time_mean" or c == "generated_answers_time_mean")]
    gen_acc_cols = [c.replace("time_mean", "acc") for c in gen_time_cols]
    # merged = merged.dropna()

    # set all none values in time columns to LARGE_NUMBER

    report_df[gen_time_cols] = report_df[gen_time_cols].fillna(LARGE_NUMBER)

    # set all none values in acc columns to 0
    report_df[gen_acc_cols] = report_df[gen_acc_cols].fillna(0)

    report_df["input_shape"] = report_df["input_stats_all"].apply(lambda x: np.array(x).shape)
    
    # for each row, find the best time among those submissions that have acc = 1
    for time_col in gen_time_cols:
        acc_col = time_col.replace("time_mean", "acc")
        
        report_df.loc[report_df[acc_col] < 1, time_col] = LARGE_NUMBER

        # also ignore solutions that have a different shape than the input
        all_stats_col = time_col.replace("time_mean", "stats_all")
        report_df.loc[report_df[all_stats_col].apply(lambda x: np.array(x).shape) != report_df["input_shape"], time_col] = LARGE_NUMBER

    report_df["best_generated_time_mean"] = report_df[gen_time_cols].min(axis=1)

    # find the column name of the best time. This tells us which of the N solutions was the best
    report_df["best_generated_time_mean_col"] = report_df.apply(
        lambda row: [col for col in gen_time_cols if row[col] == row["best_generated_time_mean"]][
            0
        ],
        axis=1,
    )

    # find the best solution
    report_df["best_generated_soln"] = report_df.apply(
        lambda row: row[row["best_generated_time_mean_col"].replace("_time_mean", "")], axis=1
    )
    report_df["best_generated_time_std"] = report_df.apply(
        lambda row: row[row["best_generated_time_mean_col"].replace("time_mean", "time_std")],
        axis=1,
    )
    report_df["best_generated_stats"] = report_df.apply(
        lambda row: row[row["best_generated_time_mean_col"].replace("time_mean", "stats")],
        axis=1,  # array / list of runs aggregates across test cases
    )

    report_df["best_tag"] = report_df["best_generated_time_mean_col"].apply(
        lambda x: re.sub(r".*time_mean_(.*)", r"\1", x)
    )

    report_df["diff"] = report_df.apply(
        lambda row: get_minimal_diff(row["input"], row["best_generated_soln"]), axis=1
    )
    report_df["diff"] = report_df["diff"].apply(lambda x: x.strip())
    report_df["diff_empty"] = report_df["diff"].apply(lambda x: "".join([d.strip() for d in x]) == "")
    report_df = report_df[~report_df["diff_empty"]]


    report_df = report_df[report_df["input"] != report_df["best_generated_soln"]]

    if len(report_df) == 0:
        return "0, 1, 1, 1"

    # report_df["p_value"] = report_df.apply(lambda row: get_welch_t_test_p(row), axis=1)

    # t = mu(input) - mu(best_generated) - X% of mu(input) / std_error (t test for a speedup of X%; Null hyp = speedup <= X%)
    report_df[f"p_value_{required_speedup}pct"] = report_df.apply(
        lambda row: get_r_ttest_p(
            row, generated_answer_field_tag="best_generated", required_speedup=required_speedup
        ),
        axis=1,
    )

    if len(report_df) == 0:
        print("No significant difference")


    report_df["speedup"] = report_df["input_time_mean"] / (
        report_df["best_generated_time_mean"] + 1e-8
    )
    report_df["speedup_vs_ref"] = report_df["input_time_mean"] / (
        report_df["reference_time_mean"] + 1e-8
    )

    # report_df["cohens_d"] = report_df.apply(
    #     lambda row: get_cohens_d(row, generated_answer_field_tag="best_generated"), axis=1
    # )

    # print(report_df["status"].apply(lambda x: sorted(x)).value_counts())
    report_df = report_df[report_df["best_generated_time_mean"] < LARGE_NUMBER]

    # report_df["diff_empty"] = report_df["diff"].apply(lambda x: len(x) == 0)
    # report_df = report_df[~report_df["diff_empty"]]

    
    # num_recs = len(report_df)
    # test_set_size = len(report_df)
    # report_df = report_df[report_df["p_value"] < 0.05]
    report_df = report_df[report_df[f"p_value_{required_speedup}pct"] < 0.05]

    # print(f" Reference speedup: {report_df[report_df['reference_acc'] == 1]['speedup_vs_ref'].mean():.2f}x")
    
    opt_pct = round(len(report_df) * 100 / test_set_size, 2)
    # generate for latex: opt & mean speedup & min speedup & max speedup
    
    # write report_df to a file where it can be used for further analysis
    ext = "cpp" if lang == "cpp" else "py"
    # add name of the file to the report_df

    analysis_report_name = f"/tmp/report_df_{lang}_{n_samples}.{os.path.basename(report_path)}.{ext}"
    write_for_analysis(report_df, analysis_report_name)
    
    mean_speedup = report_df["speedup"].mean()
    
    if default_speedup:
        
        speedups = [default_speedup for _ in range(test_set_size - len(report_df))]
        all_speedups = report_df["speedup"].tolist() + speedups
        mean_speedup = np.mean(all_speedups)
    if return_values:
        return opt_pct, mean_speedup

            
    # print(f"{opt_pct} & {report_df['speedup'].mean():.2f} & {report_df['speedup'].max():.2f}")
    return report_df[['problem_id', 'submission_id_v0', 'speedup', f"p_value_{required_speedup}pct"]]
    # return report_df[['problem_id', 'input', 'best_generated_soln', 'diff', 'p_value', 'p_value_10pct', 'speedup', 'speedup_vs_ref', 'cohens_d', 'best_tag']]


def write_for_analysis(report_df, filename):
    def _diff(code1, code2) -> str:
        # Use ndiff instead of unified_diff as it provides cleaner output and does not include meta-symbols
        diff = list(difflib.ndiff(code1.splitlines(keepends=True), code2.splitlines(keepends=True)))

        # Filter out lines without changes
        diff_minus_meta = [line for line in diff if line.startswith('+ ') or line.startswith('- ')]
        
        return "\n".join(diff_minus_meta)

    report_df = report_df.sort_values("speedup", ascending=False)
    with open(filename, "w") as f:
        for i, row in report_df.iterrows():
            f.write(f"{'-'*80}")
            # write input, output, reference, speedup and p-value, problem id
            # first print the header
            info = f"""
Problem ID: {row['problem_id']}
Speedup: {row['speedup']}
Speedup vs. reference: {row['speedup_vs_ref']}
best_generated_time_mean: {row['best_generated_time_mean']}
input_time_mean: {row['input_time_mean']}
submission_id_v0: {row['submission_id_v0']}
"""         
            f.write(info)
            f.write(f"Input:\n\n\n {row['input']}")
            f.write("\n\n\n")
            # f.write(f"Reference:\n\n\n {row['target']}")
            # f.write("\n\n\n")
            f.write(f"Best generated:\n\n\n {row['best_generated_soln']}")
            f.write("\n\n\n")
            f.write(f"Diff:\n\n\n {_diff(row['input'], row['best_generated_soln'])}")
            f.write(f"{'-'*80}")


def get_welch_t_test_p(row, n_samples: int = 25):
    p_value = scipy.stats.ttest_ind_from_stats(
        mean1=row["best_generated_time_mean"],
        std1=row["best_generated_time_std"],
        mean2=row["input_time_mean"],
        std2=row["input_time_std"],
        nobs1=n_samples,
        nobs2=n_samples,
        equal_var=False,  # Welch's t-test
        alternative="less",
    ).pvalue
    return p_value

def analyze_runs(report_df):
    # Get the unique runs from the dataframe
    unique_runs = report_df['run'].unique()

    # Initialize a list to store the results
    results = []

    # Iterate over unique runs
    for run in unique_runs:
        # Filter problems up to the current run
        problems_df = report_df[report_df['run'] <= run]

        # Drop duplicate submission_id_v0, keeping the first occurrence (previous speedup)
        problems_df = problems_df.drop_duplicates(subset='submission_id_v0', keep='first')

        # Calculate the average speedup for the current run
        avg_speedup = problems_df['speedup'].mean()

        # Append the result to the list
        results.append({'run': run, 'num_problems': len(problems_df), 'avg_speedup': avg_speedup})

    # Create a new dataframe to display the results
    results_df = pd.DataFrame(results)

    return results_df

if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("report_path", type=str, default="report.jsonl", nargs="?")
    args.add_argument("--lang", type=str, default="python")
    args.add_argument("--required_speedup", type=float, default=0.05)

    args = args.parse_args()

    reports = dict()
    for i in range(4):
        possible_report_path = f"{args.report_path}/output.pie.jsonl.{i}.report"
        # check if the file exists
        
        if os.path.exists(possible_report_path):
            reports[i] = summarize(report_path=possible_report_path, lang=args.lang, n_samples=32, required_speedup=args.required_speedup, return_values=False)
            reports[i]["run"] = i
    
    # concat all reports
    report = pd.concat([pd.DataFrame.from_dict(reports[i]) for i in reports])
    
    runs = analyze_runs(report)
    print(runs)