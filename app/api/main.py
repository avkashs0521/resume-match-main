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
    """Updates state and returns strictly compliant response format"""
    # Convert dict to Action model
    action = Action(**action_data)
    obs, reward, done, info = env.step(action)
    
    # Extract float reward - handles Case 1 (dict) & Case 2 (Pydantic object)
    if isinstance(reward, dict):
        reward_value = reward.get("score", 0.0)
        trust_val = reward.get("trust_score")
        feedback_val = reward.get("feedback")
    else:
        # Pydantic model
        reward_value = getattr(reward, "score", 0.0)
        trust_val = getattr(reward, "trust_score", None)
        feedback_val = getattr(reward, "feedback", None)

    return {
        "observation": obs,
        "reward": max(min(float(reward_value), 0.999), 0.001),  # FORCE float strictly between 0 and 1
        "done": bool(done),
        "info": {
            **(info or {}),
            "trust_score": trust_val,
            "feedback": feedback_val
        }
    }

@app.get("/state")
def state():
    """Returns current state wrapped in observation"""
    return {"observation": env.state()}

@app.get("/")
def home():
    return {"message": "OpenEnv Resume Arena API is running 🚀"}