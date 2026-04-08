from .similarity import compute_similarity
import numpy as np
from scipy.optimize import linear_sum_assignment


# ✅ EASY → best match only
def match_easy(resumes, jobs):
    sim = compute_similarity(resumes, jobs)
    # sim += np.random.normal(0, 0.02, sim.shape) # Removed randomness for deterministic baseline requirement

    matches = {}
    for i, job in enumerate(jobs):
        best_idx = np.argmax(sim[i])
        matches[job["id"]] = resumes[best_idx]["id"]

    return matches


# ✅ MEDIUM → ranking with filtering
def match_medium(resumes, jobs):
    job = jobs[0]

    sim = compute_similarity(resumes, [job])[0]
    # sim += np.random.normal(0, 0.02, sim.shape) # Removed randomness for deterministic baseline requirement

    # 🔥 SKILL BOOST
    boosted_scores = []

    for i, r in enumerate(resumes):
        text = str(r.get("text", "")).lower()
        skills = job.get("skills_required", [])

        skill_match = sum(skill.lower() in text for skill in skills)

        # boost score
        score = sim[i] + 0.4 * skill_match
        boosted_scores.append(score)

    ranked_indices = np.argsort(boosted_scores)[::-1]

    ranked_resumes = [resumes[i]["id"] for i in ranked_indices[:3]]

    return ranked_resumes

# ✅ HARD → optimal assignment
def match_hard(resumes, jobs):
    sim = compute_similarity(resumes, jobs)
    # sim += np.random.normal(0, 0.02, sim.shape) # Removed randomness for deterministic baseline requirement

    cost = -sim
    row_ind, col_ind = linear_sum_assignment(cost)

    assignments = {}
    for j, r in zip(row_ind, col_ind):
        assignments[jobs[j]["id"]] = resumes[r]["id"]

    return assignments


# ✅ UTILITY → Get top matching resumes for a job
def get_top_k(job, resumes, k=5):
    """
    Ranks all resumes against a job and returns the top K resume IDs.
    Used for shortlisting steps in multi-step simulations.
    """
    if isinstance(job, list):
        job = job[0]
        
    sim = compute_similarity(resumes, [job])[0]
    
    # Sort indices by similarity descending
    top_indices = np.argsort(sim)[::-1]
    
    return [resumes[i]["id"] for i in top_indices[:k]]


# ✅ RANDOM → baseline (for comparison)
def match_random(resumes, jobs, task_type="easy"):
    import random
    sim = compute_similarity(resumes, jobs)
    
    if task_type == "medium":
        scores = sim[0]
        top_5_idx = np.argsort(scores)[::-1][:5]
        top_5_rids = [resumes[i]["id"] for i in top_5_idx]
        return random.sample(top_5_rids, min(3, len(top_5_rids)))
    
    matches = {}
    for i, job in enumerate(jobs):
        scores = sim[i]
        top_5_idx = np.argsort(scores)[::-1][:5]
        top_5_rids = [resumes[idx]["id"] for idx in top_5_idx]
        matches[job["id"]] = random.choice(top_5_rids)
    return matches