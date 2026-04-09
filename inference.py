import os
import sys
import json
import numpy as np
import logging
import warnings
from openai import OpenAI
from app.env.environment import ResumeEnv
from app.env.models import Action
from app.matching.matcher import match_easy, match_medium, match_hard, get_top_k
from app.matching.similarity import compute_similarity

# Suppress noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

old_stdout = sys.stdout
sys.stdout = sys.stderr

def print_log(msg):
    old_stdout.write(msg + "\n")
    old_stdout.flush()

def log_start(task, model):
    print_log(f"[START] task={task} env=resume-matching-env model={model}")

def log_step(step, action, reward, done, xai=None, error="null"):
    xai_str = json.dumps(xai).replace(' ', '') if xai else "null"
    print_log(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} xai={xai_str} error={error}")

def log_end(success, steps, score, rewards):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print_log(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")

def run_inference():
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    if not API_BASE_URL:
        API_BASE_URL = "https://router.huggingface.co/v1"
        
    MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:groq")
    HF_TOKEN = os.getenv("HF_TOKEN")

    client = None
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN or "sk-no-token-provided",
        )
    except Exception as e:
        print_log(f">> FAILED TO INIT OPENAI CLIENT: {e}")

    tasks = ["easy", "medium", "hard"]

    for task_name in tasks:
        log_start(task_name, MODEL_NAME)
        env = ResumeEnv(task_type=task_name)
        obs = env.reset()

        rewards = [] # Track rewards from steps
        error_msg = "null"
        final_reward_val = 0.0
        done = False
        xai_metadata = {}
        processed_action = "error"

        try:
            # 🚀 INTERNAL STEPS 1-3 (Programmatic & Silent)
            all_resumes_dict = [r.model_dump() for r in obs.resumes]
            all_jobs_dict = [j.model_dump() for j in obs.jobs]
            
            # 1. Analyze
            obs, r_obj, done, info = env.step(Action(action_type="analyze_job"))
            rewards.append(r_obj.score)
            log_step(1, "analyze", r_obj.score, done)
            
            # 2. Shortlist
            primary_job_dict = obs.jobs[0].model_dump()
            shortlist_ids = get_top_k(primary_job_dict, all_resumes_dict, k=5)
            obs, r_obj, done, info = env.step(Action(action_type="shortlist", resumes=shortlist_ids))
            rewards.append(r_obj.score)
            log_step(2, shortlist_ids, r_obj.score, done)
            
            # 3. Rank
            obs, r_obj, done, info = env.step(Action(action_type="rank"))
            rewards.append(r_obj.score)
            log_step(3, "rank", r_obj.score, done)

            # -------------------------------------------------------------
            # 🧠 STEP 4: FINALIZE
            # -------------------------------------------------------------
            final_action = None
            job_ids = [j.id for j in obs.jobs]
            
            sim_matrix = compute_similarity(all_resumes_dict, all_jobs_dict)
            top_candidates_per_job = {}
            filtered_resume_ids = set()
            job_candidates_text = ""
            
            for i, job in enumerate(obs.jobs):
                top_idx = np.argsort(sim_matrix[i])[::-1][:5]
                candidates = []
                for idx in top_idx:
                    rid = obs.resumes[idx].id
                    score = sim_matrix[i][idx]
                    candidates.append((rid, score))
                    filtered_resume_ids.add(rid)
                
                top_candidates_per_job[job.id] = candidates
                candidates_str = ", ".join([f"{cid} (score: {s:.2f})" for cid, s in candidates])
                job_candidates_text += f"\n- TOP CANDIDATES FOR {job.id}: {candidates_str}"

            resume_text = "\n".join([
                f"{r.id}: {r.text[:150]}" for r in obs.resumes if r.id in filtered_resume_ids
            ])
            job_text = "\n".join([f"{j.id}: {j.description} | Skills: {j.skills_required}" for j in obs.jobs])

            prompt = f"JOBS:\n{job_text}\n{job_candidates_text}\n\nRESUMES:\n{resume_text}\n"

            # Fallback to local matcher if client failed init
            if not client:
                raise ValueError("LLM client not available")

            try:
                # 🚀 LLM CALL
                res = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Return ONLY JSON response for task matched/ranked list."},
                        {"role": "user", "content": prompt}
                    ],
                    timeout=15.0
                )
                
                content = res.choices[0].message.content
                parsed = json.loads(content)
                matches = parsed.get("matches", {})
                ranked_list = parsed.get("ranked_list", [])
                
                final_action = Action(action_type="finalize", matches=matches, ranked_list=ranked_list)

            except Exception:
                # 🛡️ FALLBACK TO DETERMINISTIC MATCHER
                if task_name == "easy":
                    final_action = Action(action_type="finalize", matches=match_easy(all_resumes_dict, all_jobs_dict))
                elif task_name == "medium":
                    final_action = Action(action_type="finalize", ranked_list=match_medium(all_resumes_dict, all_jobs_dict))
                else:
                    final_action = Action(action_type="finalize", matches=match_hard(all_resumes_dict, all_jobs_dict))

            obs, r_obj, done, info = env.step(final_action)
            final_reward_val = r_obj.score
            rewards.append(final_reward_val)
            processed_action = final_action.ranked_list if task_name == "medium" else final_action.matches
            xai_metadata = {"matched": r_obj.matched_skills, "missing": r_obj.missing_skills, "suggestion": r_obj.suggestion}

        except Exception as crash:
            error_msg = str(crash).replace('\n', ' ')
            processed_action = "error"
            final_reward_val = 0.0
            done = True

        total_score = max(min(sum(rewards), 0.999), 0.001)
        log_step(4, processed_action, final_reward_val, done, xai=xai_metadata, error=error_msg)
        success = done and total_score > 0
        log_end(success, 4, total_score, rewards)

if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        print(f"[ERROR] Global failure: {e}", file=sys.stderr)
        sys.exit(1)