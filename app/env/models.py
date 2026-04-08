from pydantic import BaseModel
from typing import List, Dict, Union

class Resume(BaseModel):
    id: str
    skills: List[str]
    experience: int
    text: str

class Job(BaseModel):
    id: str
    skills_required: List[str]
    description: str

class Observation(BaseModel):
    resumes: List[Resume]
    jobs: List[Job]
    current_matches: Dict[str, Union[str, List[str]]]
    step_count: int
    shortlisted_resumes: List[str] = []
    rejected_resumes: List[str] = []
    current_step_name: str = "analyze_job"
    confidence_score: float = 0.0
    trust_score: float = 1.0
    action_history: List[str] = []

class Action(BaseModel):
    action_type: str = "finalize"
    resumes: List[str] = []
    matches: Dict[str, str] = {}
    ranked_list: List[str] = []

class Reward(BaseModel):
    score: float
    trust_score: float = 1.0
    feedback: str
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    suggestion: str = ""