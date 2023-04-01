import pandas as pd

df = pd.read_json("/usr1/amadaan/shufflegen/data/original/commongen/val.jsonl", lines=True, orient="records")

from itertools import chain
all_concepts = set(chain(*df['concepts'].tolist()))

# challenging data with 10-15 concepts

import random
random.seed(42)

n_samples = 200
res = []
for i in range(n_samples):
    k = random.randint(20, 30)
    concepts = random.sample(all_concepts, k=k)
    res.append({"concepts": concepts})

pd.DataFrame(res).to_json("data/commongen_very_challenging.jsonl", lines=True, orient="records")
    
