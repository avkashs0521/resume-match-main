import pandas as pd
import json
import random
import re

df = pd.read_csv("data/raw/resume_dataset.csv")
df.columns = df.columns.str.strip()

df = df[["Resume_str", "Category"]]

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    return text

pairs = []

for _ in range(15000):  # 🔥 increased data
    r1 = df.sample(1).iloc[0]
    r2 = df.sample(1).iloc[0]

    resume1 = clean(r1["Resume_str"])
    resume2 = clean(r2["Resume_str"])

    overlap = len(set(resume1.split()) & set(resume2.split()))

    # 🔥 better labeling
    if r1["Category"] == r2["Category"] and overlap > 10:
        label = 1.0
    elif overlap > 5:
        label = 0.5
    else:
        label = 0.0

    pairs.append({
        "resume": resume1,
        "job": resume2,
        "label": label
    })

with open("data/training/train.json", "w") as f:
    json.dump(pairs, f, indent=2)

print("✅ Better training data created!")