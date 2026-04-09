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
    """Updates state and returns {observation, reward, done, info}"""
    # Convert dict to Action model
    action = Action(**action_data)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info if info is not None else {}
    }

@app.get("/state")
def state():
    """Returns current state wrapped in observation"""
    return {"observation": env.state()}

@app.get("/")
def home():
    return {"message": "OpenEnv Resume Arena API is running 🚀"}