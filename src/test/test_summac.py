# test_summac.py

import time
from summac.model_summac import SummaCZS

print("Loading model...")
t = time.time()

model = SummaCZS(
    model_name="mnli",
    granularity="sentence",
    device="cpu"
)

print(f"Loaded in {time.time()-t:.2f}s")

source = """
This is a short document.
It has only two sentences.
"""

summary = """
This is a summary.
"""

print("Scoring...")

t = time.time()

result = model.score(
    [source],
    [summary]
)

print(result)
print(f"Score time: {time.time()-t:.2f}s")