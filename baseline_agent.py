from app.env.environment import ResumeEnv
from app.env.models import Action
from app.matching.matcher import match_easy, match_medium, match_hard, get_top_k

def run_baseline_agent():
    """
    Upgraded Baseline Agent for Resume Arena.
    
    This agent demonstrates a perfect 'Golden Path' through the multi-step simulation:
    1. ANALYZE: Reviews the job requirements provided in the observation.
    2. SHORTLIST: Uses semantic similarity (TF-IDF + Embeddings) to filter top-5 candidates.
    3. RANK: Signals the environment that it is ready to formulate final decisions.
    4. FINALIZE: Executes the appropriate task-specific matcher for the final score.
    """
    tasks = ["easy", "medium", "hard"]
    
    print("====================================")
    print("🤖 UPGRADED BASELINE AGENT SIMULATION")
    print("====================================")

    for task in tasks:
        print(f"\n🚀 Starting Task: {task.upper()}")
        env = ResumeEnv(task_type=task)
        obs = env.reset()
        
        # ✅ VALIDATION CHECK: Confirm 1-based job IDs
        # print(f"DEBUG JOB IDS: {[j.id for j in obs.jobs]}")
        assert all(j.id.startswith("j") and j.id != "j0" for j in obs.jobs), "ERROR: Found non-standard or 0-based job ID!"
        
        # 0. Context Extraction
        # We extract raw dictionary data from the Pydantic models for the matcher logic
        all_resumes = [r.model_dump() for r in obs.resumes]
        active_jobs = [j.model_dump() for j in obs.jobs]
        
        # 1. Analyze Job Step
        # The agent acknowledges it has 'read' the job descriptions
        print("  -> Execution: analyze_job")
        obs, reward, done, _ = env.step(Action(action_type="analyze_job"))
        print(f"     State: {obs.current_step_name} | Reward: {reward.score}")
        
        # 2. Shortlist Step
        # Uses the matcher.get_top_k utility to select 5 candidates instead of random slicing
        primary_job = active_jobs[0]
        shortlist_ids = get_top_k(primary_job, all_resumes, k=5)
        
        print(f"  -> Execution: shortlist {shortlist_ids[:3]}...")
        obs, reward, done, _ = env.step(Action(action_type="shortlist", resumes=shortlist_ids))
        print(f"     State: {obs.current_step_name} | Reward: {reward.score}")

        # 3. Rank Step
        # Signals that shortlisting is complete and it is ready to rank/assign
        print("  -> Execution: rank")
        obs, reward, done, _ = env.step(Action(action_type="rank"))
        print(f"     State: {obs.current_step_name} | Reward: {reward.score}")

        # 4. Finalize assignments
        # Uses task-specific deterministic matchers to produce the optimal final action
        print("  -> Execution: finalize")
        
        # 🔍 DEBUG JOB IDS
        # print("DEBUG JOB IDS:", [j.id for j in obs.jobs])
        
        action = None
        if task == "easy":
            # Direct 1:1 match using environment-provided job ID
            job = obs.jobs[0]
            matches_data = match_easy(all_resumes, active_jobs)
            resume_id = matches_data.get(job.id)
            
            matches = {job.id: resume_id}
            print(f"     Match Chosen: {matches}")
            action = Action(action_type="finalize", matches=matches)
        elif task == "medium":
            # Ordered top-3 list
            ranked_list = match_medium(all_resumes, active_jobs)
            print(f"     Ranked List: {ranked_list}")
            action = Action(action_type="finalize", ranked_list=ranked_list)
        else:
            # Batch optimization for HARD task using zip as requested
            matches_data = match_hard(all_resumes, active_jobs)
            
            assignments = {}
            for job in obs.jobs:
                assignments[job.id] = matches_data.get(job.id)
            
            print(f"     Assignments: {assignments}")
            action = Action(action_type="finalize", matches=assignments)
        
        # Terminal step executes the final reward calculation
        obs, reward, done, _ = env.step(action)
        print(f"     State: Final | Reward: {reward.score} | Done: {done}")
        
        # Log XAI insights for debug visibility
        if reward.matched_skills:
            print(f"     XAI: Matched={reward.matched_skills}")
        if reward.missing_skills:
            print(f"     XAI: Missing={reward.missing_skills}")

if __name__ == "__main__":
    run_baseline_agent()
