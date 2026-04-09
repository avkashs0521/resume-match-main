from fastapi import FastAPI
from app.env.environment import ResumeEnv
from app.env.models import Action

app = FastAPI(title="OpenEnv Resume Arena")

# Initialize environment
env = ResumeEnv(task_type="medium")

@app.post("/reset")
def reset():
    """Initializes environment and returns wrapped observation"""
    obs = env.reset()
    return {"observation": obs}

@app.post("/step")
def step(action_data: dict):
    """Updates state and returns strictly OpenEnv-compliant response format"""
    # Convert dict to Action model for env compatibility
    action = Action(**action_data)
    obs, reward, done, info = env.step(action)
    
    # Get float reward (handling both Pydantic model and potential dict)
    reward_score = reward.score if hasattr(reward, "score") else (reward.get("score") if isinstance(reward, dict) else reward)
    
    # Prepare info by merging existing info and extra reward metadata
    response_info = (info or {})
    if hasattr(reward, "trust_score"):
        response_info.update({
            "trust_score": reward.trust_score,
            "feedback": reward.feedback
        })
    elif isinstance(reward, dict):
        response_info.update({
            "trust_score": reward.get("trust_score"),
            "feedback": reward.get("feedback")
        })

    return {
        "observation": obs,
        "reward": float(reward_score),
        "done": done,
        "info": response_info
    }

@app.get("/state")
def state():
    """Returns current state wrapped in observation"""
    return {"observation": env.state()}

@app.get("/")
def home():
    return {"message": "OpenEnv Resume Arena API is running 🚀"}