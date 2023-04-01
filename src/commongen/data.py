class AcronymInitPrompts:

    prompt: str = """Title: A Survey of Active Network Research

Acronym: SONAR

###

Title: A Scalable, Commutative Replica Dictatorship for Practical Optimistic Replication

Acronym: SCRATCHPAD

### 

Title: Blockchain: A Peer-to-Peer Electronic Cash System

Acronym: BTC

###

Title: A Taxonomy of DDoS Attacks and DDoS Defense Mechanisms

Acronym: TDDM

###

Title: Bidirectional Encoder Representations from Transformers

Acronym: BERT

###

Title: Sequence to Sequence Learning with Neural Networks

Acronym: Seq2Seq

###

Title: Densely Connected Convolutional Networks for Image Classification

Acronym: DenseNet

###

Title: A Dynamic Programming Algorithm for RNA Secondary Structure Prediction

Acronym: DYNALIGN

###

Title: Fast Parallel Algorithms for Short-Range Molecular Dynamics

Acronym: FASTMD

###

Title: Real-Time Collaborative Editing Systems

Acronym: COCOON

###

Title: Efficient Data Structures for Large Scale Graph Processing

Acronym: EDGE

###

"""
import re
import pandas as pd

def acronym_init_prompts_to_tsv():
    prompt = AcronymInitPrompts.prompt
    
    examples = re.split(r"\n###\s*\n", prompt)
    res = []
    for example in examples:
        try:
            title, acronym = re.split(r"\nAcronym: ", example)
            title = re.sub(r"Title: ", "", title)
            res.append({
                "title": title.strip(),
                "acronym": acronym.strip()
            })
        except:
            pass
    df = pd.DataFrame(res)
    df.to_json("data/prompt/acronym/init.jsonl", orient="records", lines=True)
    
        

class AcronymFeedbackPrompt:
    prompt: str = """We want to score each acronym on five qualities: i) ease of pronunciation, ii) ease of spelling, and iii) relation to the title, iv) positive connotation, v) well-known.

Here are some examples of this scoring rubric:

Title: "The Effect of the Internet on the Quality of Life of the Elderly"

Acronym: TEOTIOTQOLE

Scores:

* Ease of pronunciation: TEOTIOTQOLE is pronounced as TEO-TEE-OT-TEE-OT-QO-LEE. This is a very difficult acronym to pronounce. 1/5
* Ease of spelling: TEOTIOTQOLE is a very difficult acronym to spell. 1/5
* Relation to title: TEOTIOTQOLE has no relation to the title. 1/5
* Positive connotation: TEOTIOTQOLE is meaningless, thus it has no positive connotation. 1/5
* Well-known: TEOTIOTQOLE is not a well-known acronym. 1/5

* Total score: 5/25

###

Title: "The Effect of the Internet on the Quality of Life of the Elderly"

Acronym: EIQOLE

Scores:

* Ease of pronunciation: EIQOLE is pronounced "eye-cowl". This is easy acronym to pronounce. 3/5
* Ease of spelling: Hard to remember the "QO" in the middle. 2/5
* Relation to title: EIQOLE has no relation to the title. It is just a random acronym. 1/5
* Positive connotation: EIQOLE is not a positive acronym. It is just a random acronym. 1/5
* Well-known: EIQOLE is not a well-known acronym. 1/5

* Total score: 8/25

###

Title: "The Effect of the Internet on the Quality of Life of the Elderly"

Acronym: EQULE

Scores:

* Ease of pronunciation: EQULE is pronounced "eye-cue-lee". This is easy acronym to pronounce. 4/5
* Ease of spelling: EQULE is easy to spell. 3/5
* Relation to title: EQULE sounds like "equal" which is related to the title. 5/5
* Positive connotation: EQULE is a positive acronym. 5/5
* Well-known: EQULE is close to the word "equal" which is a well-known word. 4/5

* Total score: 19/25

###

Title: "Blockchain: A Peer-to-Peer Electronic Cash System"

Acronym: BAP2PESCS

Scores:

* Ease of pronunciation: BAP2PESCS is pronounced "bee-ay-pee-two-pee-ess-see-ess-see". This is a very difficult acronym to pronounce. 1/5
* Ease of spelling: BAP2PESCS is a very difficult acronym to spell. 1/5
* Relation to title: BAP2PESCS has no relation to the title. 1/5
* Positive connotation: BAP2PESCS is meaningless, thus it has no positive connotation. 1/5
* Well-known: BAP2PESCS is not a well-known acronym. 1/5

* Total score: 5/25

###

Title: "Blockchain: A Peer-to-Peer Electronic Cash System"

Acronym: BAP2PES

Scores:

* Ease of pronunciation: BAP2PES is pronounced "bap-to-pes". This is easier acronym to pronounce. 2/5
* Ease of spelling: BAP2PES is not too easy to spell. 2/5
* Relation to title: BAP2PES has no relation to the title. It is just a random acronym. 1/5
* Positive connotation: BAP2PES is not a positive acronym. It is just a random acronym. 1/5
* Well-known: BAP2PES is not a well-known acronym. 1/5

* Total score: 7/25

###

Title: "Blockchain: A Peer-to-Peer Electronic Cash System"

Acronym: BECash

Scores:

* Ease of pronunciation: BECash is pronounced "bee-cash". This is easy acronym to pronounce. 4/5
* Ease of spelling: BECash is easy to spell. 4/5
* Relation to title: BECash mentions "cash" which is somewhat related to the title. 3/5
* Positive connotation: BECash is a positive acronym. 5/5
* Well-known: BECash is close to the word "cash" which is a well-known word. 4/5

* Total score: 20/25

###

Title: "Sequence to Sequence Learning with Neural Networks"

Acronym: STSLWN

Scores:

* Ease of pronunciation: STSLWN is pronounced "ess-tee-ess-ell-double-you-enn". This is a very difficult acronym to pronounce. 1/5
* Ease of spelling: STSLWN is a very difficult acronym to spell. 1/5
* Relation to title: STSLWN has no relation to the title. 1/5
* Positive connotation: STSLWN is meaningless, thus it has no positive connotation. 1/5
* Well-known: STSLWN is not a well-known acronym. 1/5

* Total score: 5/25

###

Title: "Sequence to Sequence Learning with Neural Networks"

Acronym: STSLN

Scores:

* Ease of pronunciation: STSLN is pronounced "ess-tee-ess-ell-en". This is easier acronym to pronounce. 2/5
* Ease of spelling: STSLN is not too easy to spell. 2/5
* Relation to title: STSLN has no relation to the title. It is just a random acronym. 1/5
* Positive connotation: STSLN is not a positive acronym. It is just a random acronym. 1/5
* Well-known: STSLN is not a well-known acronym. 1/5

* Total score: 7/25

###

Title: "Sequence to Sequence Learning with Neural Networks"

Acronym: Seq2Seq

Scores:

* Ease of pronunciation: Seq2Seq is pronounced "seq-two-seq". This is easy acronym to pronounce. 4/5
* Ease of spelling: Seq2Seq is easy to spell. 4/5
* Relation to title: Seq2Seq mentions "sequence" which is somewhat related to the title. 3/5
* Positive connotation: Seq2Seq is a positive acronym. It gives out a sense of ease with which the learning algorithm can be used. 5/5
* Well-known: Seq2Seq is close to the word "sequence" which is a well-known word. 4/5

* Total score: 20/25

###

"""


def acronym_iterate_prompt_to_tsv():
    
    import re
    import pandas as pd
    prompt = AcronymFeedbackPrompt.prompt
    
    res = []
    examples = prompt.split("###")
    for example in examples:
        try:
            if not example:
                continue
            example = example.strip()

            # title is everything from "Title: " to the next newline
            title = re.search(r"Title: (.*)\n", example).group(1)
            acronym = re.search(r"Acronym: (.*)\n", example).group(1)
            ease_of_pronunciation = re.search(r"Ease of pronunciation: (.*)\n", example).group(1)
            ease_of_spelling = re.search(r"Ease of spelling: (.*)\n", example).group(1)
            relation_to_title = re.search(r"Relation to title: (.*)\n", example).group(1)
            positive_connotation = re.search(r"Positive connotation: (.*)\n", example).group(1)
            well_known = re.search(r"Well-known: (.*)\n", example).group(1)
            total_score = re.search(r"Total score: (.*)", example).group(1)
            res.append(
                {
                    "title": title.replace("\"", ""),
                    "acronym": acronym.replace("\"", ""),
                    "pronunciation_score": ease_of_pronunciation.replace("\"", ""),
                    "spelling_score": ease_of_spelling.replace("\"", ""),
                    "relation_score": relation_to_title.replace("\"", ""),
                    "connotation_score": positive_connotation.replace("\"", ""),
                    "well_known_score": well_known.replace("\"", ""),
                    "total_score": total_score.replace("\"", ""),
                }
            )
        except Exception as e:
            print(e)
    df = pd.DataFrame(res)
    print(df)
    df.to_json("data/prompt/acronym/feedback.jsonl", orient="records", lines=True)

if __name__ == "__main__":
    acronym_init_prompts_to_tsv()
    acronym_iterate_prompt_to_tsv()