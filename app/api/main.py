from fastapi import FastAPI
from app.env.environment import ResumeEnv
from app.env.models import Action

app = FastAPI(title="OpenEnv Resume Arena")

# Initialize environment
# We use a single instance for simplicity as per requirement
env = ResumeEnv(task_type="medium")

@app.post("/reset")
def reset():
    """Initializes environment and returns observation"""
    obs = env.reset()
    return obs

@app.post("/step")
def step(action_data: dict):
    """Updates state properly and returns {observation, reward, done, info}"""
    # Convert dict to Action model to ensure compatibility with environment logic
    action = Action(**action_data)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    """Returns current state"""
    return env.state()

@app.get("/")
def home():
    return {"message": "OpenEnv Resume Arena API is running 🚀"}