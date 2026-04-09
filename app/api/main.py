from fastapi import FastAPI
from app.env.environment import ResumeEnv
from app.env.models import Action

app = FastAPI(title="OpenEnv Resume Arena")

# Initialize environment
env = ResumeEnv(task_type="medium")

@app.post("/reset")
def reset():
    """Initializes environment and returns strictly wrapped observation"""
    obs = env.reset()
    return {"observation": obs}

@app.post("/step")
def step(action_data: dict):
    """Updates state and returns strictly compliant response format"""
    # Convert dict to Action model for env compatibility
    action = Action(**action_data)
    obs, reward_obj, done, info = env.step(action)
    
    # OpenEnv requires reward to be a float, NOT an object
    return {
        "observation": obs,
        "reward": float(reward_obj.score),  # Strictly float
        "done": done,
        "info": {
            "trust_score": reward_obj.trust_score,
            "feedback": reward_obj.feedback,
            "matched_skills": reward_obj.matched_skills,
            "missing_skills": reward_obj.missing_skills,
            "suggestion": reward_obj.suggestion
        }
    }

@app.get("/state")
def state():
    """Returns current state wrapped in observation"""
    return {"observation": env.state()}

@app.get("/")
def home():
    return {"message": "OpenEnv Resume Arena API is running 🚀"}