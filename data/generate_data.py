import json
import random

skills_pool = [
    "python", "java", "ml", "dl", "react",
    "node", "sql", "aws", "docker", "kubernetes"
]

def generate_resume(i):
    skills = random.sample(skills_pool, k=3)
    return {
        "id": f"r{i}",
        "skills": skills,
        "experience": random.randint(0, 5),
        "text": f"Experienced in {' '.join(skills)}"
    }

def generate_job(i):
    skills = random.sample(skills_pool, k=3)
    return {
        "id": f"j{i}",
        "skills_required": skills,
        "description": f"Looking for {' '.join(skills)} developer"
    }

def create_ground_truth(resumes, jobs):
    gt = {}

    for job in jobs:
        best = None
        max_overlap = 0

        for res in resumes:
            overlap = len(set(res["skills"]) & set(job["skills_required"]))
            if overlap > max_overlap:
                max_overlap = overlap
                best = res["id"]

        gt[job["id"]] = best

    return gt

def main():
    resumes = [generate_resume(i) for i in range(20)]
    jobs = [generate_job(i) for i in range(5)]

    gt = create_ground_truth(resumes, jobs)

    with open("resumes.json", "w") as f:
        json.dump(resumes, f, indent=2)

    with open("jobs.json", "w") as f:
        json.dump(jobs, f, indent=2)

    with open("ground_truth.json", "w") as f:
        json.dump({
            "easy": {list(gt.keys())[0]: list(gt.values())[0]},
            "medium": {list(gt.keys())[0]: list(gt.values())[:3]},
            "hard": gt
        }, f, indent=2)

    print("✅ Dataset generated!")

if __name__ == "__main__":
    main()