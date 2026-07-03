from datasets import load_dataset
multi_lexsum = load_dataset(
    "allenai/multi_lexsum",
    name="v20230518"
)

sample = multi_lexsum["test"][0]

print(sample["id"])

print(len(sample["sources"]))

print(sample["summary/long"][:500])