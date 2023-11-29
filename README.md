# Self-Refine: Iterative Refinement with Self-Feedback

_With Self-Refine, LLMs can generate feedback on their work, use it to improve the output, and repeat this process._

![image](https://raw.githubusercontent.com/madaan/self-refine/main/docs/static/images/animation_oldstyle_oneloop.gif)


<center><h4> <a href="https://selfrefine.info"> Website </a> | <a href="https://arxiv.org/pdf/2303.17651.pdf">Paper</a> </h4></center>
<hr>




<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Updates](#updates)
- [Setup](#setup)
- [Getting Started with Acronym Generation](#getting-started-with-acronym-generation)
- [Dialogue Response Generation](#dialogue-response-generation)
- [Code Readability Improvement](#code-readability-improvement)
- [Commongen](#commongen)
- [GSM-8k](#gsm-8k)
- [Yelp](#yelp)
- [PIE](#pie)
- [General setup](#general-setup)
- [Citation](#citation)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<hr>

## Updates

- **Nov 2023**: Added [visual self-refine examples](docs/visual_self_refine_examples/) and [colabs](colabs/Visual-Self-Refine-GPT4V.ipynb). Use GPT4-V to write tikz code for diagrams, and improve them iteratively.

<p align="center">
  <strong>Stokes' Theorem Example</strong><br>
  <img src="docs/visual_self_refine_examples/stokes__theorem.gif" alt="Visual Self-Refine Example 1" width="300" height="300" />
</p>
<p align="center">
  <strong>Unicorn Example</strong><br>
  <img src="docs/visual_self_refine_examples/unicorn.gif" alt="Visual Self-Refine Example 2" width="300" height="300" />
</p>



## Setup

* We use [prompt-lib](https://github.com/reasoning-machines/prompt-lib) for querying LLMs. You can install `prompt-lib` by running the following commands:

```sh
git clone https://github.com/reasoning-machines/prompt-lib
pip install prompt-lib/
```

Depending on your default settings, you may want to use the following to set a PYTHONPATH:

```sh
export PYTHONPATH=".:../:.:src:../:../../:.:prompt-lib"
```

## Getting Started with Acronym Generation


```sh
python -u src/acronym/run.py "Using language models of code for few-shot commonsense"
```

* Since we sample with temperature 0.7, you'll get different results each time you run the above command.




```swift
0 INIT> Using language models of code for few-shot commonsense

0 GEN> CLoCK
0 SCORES> * Ease of pronunciation: CLoCK is pronounced "clah-k". This is easy acronym to pronounce. 4/5 
* Ease of spelling: CLoCK is easy to spell. 5/5 
* Relation to title: CLoCK stands for "Code Language Models of Commonsense Knowledge" which is related to the title. 5/5 
* Positive connotation: CLoCK is a positive acronym. It implies accuracy and efficiency. 5/5 
* Well-known: CLoCK is not a well-known acronym. 2/5

* Total score: 21/25

1 GEN> CMNSC
1 SCORES> * Ease of pronunciation: CMNSC is pronounced "see-min-es-see". This is a difficult acronym to pronounce. 1/5 
* Ease of spelling: CMNSC is difficult to spell. 1/5 
* Relation to title: CMNSC stands for "Commonsense" which is related to the title. 4/5 
* Positive connotation: CMNSC is a neutral acronym. 3/5 
* Well-known: CMNSC is not a well-known acronym. 1/5

* Total score: 10/25

2 GEN> COMMIT
2 SCORES> * Ease of pronunciation: COMMIT is pronounced "com-mit". This is an easy acronym to pronounce. 4/5
* Ease of spelling: COMMIT is easy to spell. 5/5
* Relation to title: COMMIT stands for "Commonsense Model of Code for Intuitive Training" which is related to the title. 5/5 
* Positive connotation: COMMIT is a positive acronym. It implies commitment and dedication. 5/5
* Well-known: COMMIT is not a well-known acronym. 2/5

* Total score: 21/25
```

<hr>


## Dialogue Response Generation

```sh
PYTHONPATH="." python -u src/responsegen/run.py --output <OUTPUT FILE> --size <DATA SIZE>
```

- Use size 0 for running on all test instances

<hr>


## Code Readability Improvement


* Note: Please unzip 'data/tasks/codeclean/code_readability/codenet-python-train.jsonl.zip' before running the following commands!

- Running:
```sh
PYTHONPATH="." python -u src/readability/readability.py --output <OUTPUT FILE>
```

- Evaluation:
```sh
PYTHONPATH="." python -u src/readability/{count_comment|count_function|count_meaningful_var}.py --file <INPUT FILE>
```


<hr>


## Commongen

* We use a hard version of commongen. The data is located in `data/prompt/commongen`. You can download the data by running the following commands:

```sh
python -u src/commongen/run.py cmd stair bubble team dryer puppy aliens cat 
```

<hr>


## GSM-8k


- To run the GSM-8k task:

```sh
python -u src/gsm/run.py 
```

- The outputs will be saved in `data/tasks/gsm/gsm_outputs.jsonl`


- To evaluate the outputs:

```sh
python src/gsm/gsm_selfref_eval.py --path  data/tasks/gsm/gsm_outputs.jsonl
```

- The evaluation script will also generate a report (`data/tasks/gsm/gsm_outputs.jsonl.reports.txt`) showing examples of wrong generations, feedback, and refined feedback generations.

<hr>



## Yelp

- To run the Yelp task:

```sh
python -u src/sentiment_transfer_sr/run.py data/tasks/yelp/yelp-extreme.jso
nl 4 none
```


- The outputs will be saved in `data/tasks/yelp/`


<hr>

## PIE

- To run the PIE task:

```sh
python -u src/pie/run.py --slow_programs_file data/tasks/pie/codenet-python-test-1k.jsonl --max_attempts 4 --outfile data/tasks/pie/output --feedback_type rich
```

- For evaluation details, please see [docs/pie_eval.md](docs/pie_eval.md).

<hr>

## General setup

* Each task has three different types of prompts:

1. `Init`: used to initialize the task. This is how the initial output is generated.

2. `Feedback`: used to get feedback from the model on the intermediate results.

3. `Iterate`: used to get the next iteration from the model, based on the feedback.

* Every task has a `run.py` that initializes the prompts and runs the task.

* As an example, the prompts for commongen are as follows:

1. Init prompt:

```sh
python src/commongen/task_init.py
```

2. Feedback prompt:

```sh
 python src/commongen/feedback.py
```

3. Iterate prompt:

```sh
python src/commongen/task_iterate.py
```

You can also see these prompts on our [website](https://selfrefine.info).




## Citation

```sql
@misc{madaan2023selfrefine,
      title={Self-Refine: Iterative Refinement with Self-Feedback}, 
      author={Aman Madaan and Niket Tandon and Prakhar Gupta and Skyler Hallinan and Luyu Gao and Sarah Wiegreffe and Uri Alon and Nouha Dziri and Shrimai Prabhumoye and Yiming Yang and Sean Welleck and Bodhisattwa Prasad Majumder and Shashank Gupta and Amir Yazdanbakhsh and Peter Clark},
      year={2023},
      eprint={2303.17651},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```mermaid
flowchart LR
    Generator -->|Initializes| Unrefined
    Critic_1 --> Critique_fb
    ... --> Critique_fb
    Critic_k --> Critique_fb
    Critique_fb --> Unrefined{Output to Refine}
    Unrefined --> Refiner
    Refiner --> |R: y_t, x, fb| Refined_Output{Refined Output}
    Refined_Output --> |Stopping Criteria Not Met| Unrefined
```
