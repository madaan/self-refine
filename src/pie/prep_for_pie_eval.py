import argparse
import pandas as pd

"""Parses the self-refine outputs, extracts the output code from each attempt in a new column, and saves the results to a JSON file."""

def extract_attempt_codes(self_refine_output_path,
                          flattened_output_path, num_attempts):
    """This function creates a file where each attempt/output at each step is stored in a new column: attempt_0_code, attempt_1_code, etc.

    Args:
        input_file (_type_): _description_
        output_file (_type_): _description_
        num_attempts (_type_): _description_
    """
    outputs = pd.read_json(self_refine_output_path, orient="records", lines=True)
    rows = []

    for _, row in outputs.iterrows():
        # Convert the row to a dictionary.
        tmp = row.to_dict()
        # Extract the code from each attempt and store it in the temporary dictionary.
        for i in range(num_attempts):
            if len(row["run_logs"]) <= i:
                tmp[f"attempt_{i}_code"] = ""
            else:
                tmp[f"attempt_{i}_code"] = row["run_logs"][i]["fast_code"]

        rows.append(tmp)
    # Convert the rows list to a DataFrame and save it to a JSON file.
    pd.DataFrame(rows).to_json(flattened_output_path, orient="records", lines=True)

# Main execution of the script.
if __name__ == "__main__":
    # Initialize argument parser.
    parser = argparse.ArgumentParser(description="Generate Yaml and Extract Codes")
    # Define expected arguments.
    parser.add_argument("model", type=str, help="Model name")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file")
    parser.add_argument("base_config_path", type=str, help="Base path for config files")
    parser.add_argument("base_output_path", type=str, help="Base path for output files")
    parser.add_argument("--num_attempts", type=int, default=4, help="Number of attempts")
    # Parse provided arguments.
    args = parser.parse_args()
    
    # Construct the output file path and extract the codes.
    output_path = f"{args.base_output_path}/{args.model}/output.attempt_codes"
    extract_attempt_codes(
        self_refine_output_path=args.input_file,
        flattened_output_path=output_path,
        num_attempts=args.num_attempts,
    )
