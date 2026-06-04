from datasets import load_dataset

print("Loading Multi-LexSum")

dataset= load_dataset("allenai/multi_lexsum",name="v20230518",streaming=True,trust_remote_code=True)

print("Dataset loaded")
print(dataset)