# Instructions for evaluating runtime for PIE experiments

*TLDR: From the self-refine outputs, create a flattened version of the outputs, and then use the PIE repo to evaluate the runtime and get a report. Parse the report using `src/pie/pie_eval.py`.*

1. **Step 1** (construct yaml): For evaluating runtime for PIE experiments, we need a yaml file that contains information about the dataset, the model outputs, and the reference file. Note that self-refine generates outputs in a slightly different format. While Self-Refine generates the outputs in an array (one version per refinement step), the evaluation requires the program to be present in a single column as a script. You can optionally use [https://github.com/madaan/self-refine/tree/main/src/pie](prep_for_pie_eval.py) for this. `prep_for_pie_eval.py` creates a single file where the output from the i^th step is present in the `attempt_i_code` column. The following is an example for evaluating the initial output (`y0`).

- See `data/tasks/pie/gpt4_outputs_self_refine.jsonl` and `data/tasks/pie/gpt4_outputs_flattened.jsonl` for examples of the outputs from self-refine and the flattened version, respectively.


```
inputs_outputs_basepath: "data/codenet/generated_test_cases/"
reference_file_path: "data/tasks/pie/codenet-python-test-1k.jsonl"
num_problems_to_evaluate: -1
num_trials: 10
ignore_first_k: 0
max_time_per_run: 10
temp_dir: null
model_generated_potentially_faster_code_col: "attempt_0_code"
slow_code_col: "input"
reference_code_col: "target"
is_prompt_based: false
language: python
return_if_acc_below: 1.0
num_processes: 60
cpu_number: 15
model_generated_outputs_path: "where are the outputs we want to evaluate?"
output_report_file_path: "Where should the report file be generated?"
```

- Please see the [pie repo](https://github.com/madaan/pie-perf/blob/main/README.md#evaluating-your-method) for more details. Note that we are using generated test cases, which are also available at [pie repo](https://github.com/madaan/pie-perf/blob/main/README.md#evaluating-your-method).


2. **Step 2** (run pie eval)

Using the yaml file generated in the above step, please use the [evaluating your method](https://github.com/madaan/pie-perf/blob/main/README.md#evaluating-your-method) field to evaluate the outputs. If you run self-refine for 4 timesteps, you would create 4 yaml files and run this evaluation four times, once for each timestep. See `data/tasks/pie/gpt4_outputs.zip` for the 4 yaml files and the reports from these steps.

3. **Step 3** (parse reports and aggregate results) After the evaluation, the report is saved in `output_report_file_path.` Then, you can use `src/pie/pie_eval.py` to aggregate the results. 

### Sample outputs

- Sample yaml files for each of the 4 steps, and the corresponding outputs are located at `data/tasks/pie/gpt4_outputs.zip'.
