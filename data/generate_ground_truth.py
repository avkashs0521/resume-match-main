import json
import numpy as np
from app.matching.similarity import compute_similarity

# load data
with open("data/resumes.json") as f:
    resumes = json.load(f)

with open("data/jobs.json") as f:
    jobs = json.load(f)

gt = {
    "easy": {},
    "medium": {},
    "hard": {}
}

# compute similarity
sim = compute_similarity(resumes, jobs)

for j_idx, job in enumerate(jobs):
    job_id = job["id"]
    scores = sim[j_idx]

    ranked_indices = np.argsort(scores)[::-1]

    top_resumes = [resumes[i]["id"] for i in ranked_indices[:3]]

    # easy → best
    gt["easy"][job_id] = top_resumes[0]

    # medium → top 3
    gt["medium"][job_id] = top_resumes

    # hard → best (assignment-style)
    gt["hard"][job_id] = top_resumes[0]

# save
with open("data/ground_truth.json", "w") as f:
    json.dump(gt, f, indent=2)

print("✅ Ground truth generated!")