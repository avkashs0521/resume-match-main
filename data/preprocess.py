import pandas as pd
import re

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    return text

resumes = pd.read_csv("data/raw/resumes.csv")
jobs = pd.read_csv("data/raw/jobs.csv")

resumes["text"] = resumes.iloc[:, 1].apply(clean)
jobs["description"] = jobs.iloc[:, 1].apply(clean)

resumes.to_csv("data/processed/resumes_clean.csv", index=False)
jobs.to_csv("data/processed/jobs_clean.csv", index=False)

print("✅ cleaned")