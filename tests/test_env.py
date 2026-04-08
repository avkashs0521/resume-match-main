import pytest
from app.env.environment import ResumeEnv
from app.env.models import Action, Observation

def test_pydantic_models():
    a = Action(matches={"j1": "r1"})
    assert "j1" in a.matches
    assert isinstance(a.ranked_list, list)

def test_env_easy():
    env = ResumeEnv(task_type="easy")
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert len(obs.jobs) == 1
    
    obs, reward, done, info = env.step(Action(matches={obs.jobs[0].id: "r0"}))
    assert done == True
    assert isinstance(reward.score, float)
    assert 0.0 <= reward.score <= 1.0

def test_env_medium():
    env = ResumeEnv(task_type="medium")
    obs = env.reset()
    assert len(obs.jobs) == 1
    
    obs, reward, done, info = env.step(Action(ranked_list=["r1", "r2", "r3"]))
    assert done == True
    assert isinstance(reward.score, float)
    assert 0.0 <= reward.score <= 1.0

def test_env_hard_multi_step():
    env = ResumeEnv(task_type="hard")
    obs = env.reset()
    assert len(obs.jobs) == 5
    
    # Step 1: match 1 job
    j1 = obs.jobs[0].id
    obs, reward, done, info = env.step(Action(matches={j1: "r0"}))
    assert not done  # Should not be done yet
    assert len(obs.current_matches) == 1
    
    # Step 2: match the rest
    remaining_matches = {j.id: "r1" for j in obs.jobs[1:]}
    obs, reward, done, info = env.step(Action(matches=remaining_matches))
    assert done  # Should be done now that all jobs are matched

def test_invalid_action():
    env = ResumeEnv(task_type="easy")
    env.reset()
    
    class BadAction:
        pass
        
    obs, reward, done, info = env.step(BadAction())
    assert reward.score < 0.0
    assert done == True
