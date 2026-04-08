from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from app.matching.matcher import match_medium
from app.analysis.feedback import generate_feedback
from app.env.environment import ResumeEnv

app = FastAPI(title="Resume Matching API")


# -------- Request Schema --------
class ResumeRequest(BaseModel):
    resume_text: str


# -------- Load environment --------
env = ResumeEnv(task_type="medium")
obs = env.reset()

resumes = [r.model_dump() for r in obs.resumes]
jobs = [j.model_dump() for j in obs.jobs]


# -------- API Endpoint --------
@app.post("/match")
def match_resume(req: ResumeRequest):
    # create new resume input
    new_resume = {
        "id": "user_resume",
        "text": req.resume_text
    }

    all_resumes = resumes + [new_resume]

    # get ranking
    ranked_ids = match_medium(all_resumes, jobs)

    # 🔥 convert IDs → full resume objects
    ranked_resumes = []
    for r_id in ranked_ids:
        resume_obj = next(r for r in all_resumes if r["id"] == r_id)
        
        ranked_resumes.append({
            "id": resume_obj["id"],
            "text": resume_obj["text"][:200]  # preview (first 200 chars)
        })

    job = jobs[0]

    # feedback for user resume
    feedback = generate_feedback(new_resume, job)

    return {
    "top_matches": ranked_resumes,
    "feedback": feedback
}
@app.get("/")
def home():
    return {"message": "Resume Matching API is running 🚀"}