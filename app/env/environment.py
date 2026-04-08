import json
import os
import copy
from app.env.models import Observation, Reward
from app.env.reward import compute_reward


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def load_json(path):
    with open(os.path.join(BASE_DIR, path)) as f:
        return json.load(f)


class ResumeEnv:

    def __init__(self, task_type="easy"):
        self.task_type = task_type
        self.step_count = 0

        self.resumes = load_json("data/resumes.json")
        self.all_jobs = load_json("data/jobs.json")
        self.gt_all = load_json("data/ground_truth.json")

        self.gt = self.gt_all.get(task_type, {})
        self.current_matches = {}
        
        self.shortlisted_resumes = []
        self.rejected_resumes = []
        self.current_step_name = "analyze_job"
        self.confidence_score = 1.0
        self.trust_score = 1.0
        self.previous_step_score = 0.0
        self.action_history = []

        # Limit jobs based on task type to avoid overwhelming the LLM
        if self.task_type == "easy":
            self.jobs = [self.all_jobs[0]]
            self.gt = {self.jobs[0]["id"]: self.gt.get(self.jobs[0]["id"])}
        elif self.task_type == "medium":
            self.jobs = [self.all_jobs[0]]
            self.gt = {self.jobs[0]["id"]: self.gt.get(self.jobs[0]["id"])}
        else: # hard
            # Give 5 jobs
            self.jobs = self.all_jobs[:5]
            self.gt = {j["id"]: self.gt.get(j["id"]) for j in self.jobs}

    def reset(self):
        self.step_count = 0
        self.current_matches = {}
        self.shortlisted_resumes = []
        self.rejected_resumes = []
        self.current_step_name = "analyze_job"
        self.confidence_score = 1.0
        self.trust_score = 1.0
        self.previous_step_score = 0.0
        self.action_history = []

        return Observation(
            resumes=self.resumes,
            jobs=self.jobs,
            current_matches={},
            step_count=0,
            shortlisted_resumes=[],
            rejected_resumes=[],
            current_step_name="analyze_job",
            confidence_score=1.0,
            trust_score=1.0,
            action_history=[]
        )

    def step(self, action):
        self.step_count += 1
        done = False
        raw_reward = 0.0
        feedback = "step completed"

        # Maintain backwards compatibility: default to "finalize" if missing
        act_type = getattr(action, "action_type", "finalize")
        self.action_history.append(act_type)

        if self.step_count > 10:
            return self._finalize_step(done=True, reward=0.0, feedback="max steps reached")

        # -------------------------------------
        # DECISION ROUTING & REWARD SHAPING
        # -------------------------------------
        step_bonus = 0.0
        if act_type == "analyze_job":
            if self.current_step_name != "analyze_job":
                step_bonus -= 0.2 # Penalty for invalid order
                self.trust_score = max(0.5, self.trust_score - 0.1)
                feedback = "invalid step order - penalized"
            else:
                step_bonus += 0.01 # Minimal reasoning reward
                self.trust_score = min(1.0, self.trust_score + 0.05)
                self.current_step_name = "shortlist"

        elif act_type == "shortlist":
            if self.current_step_name not in ["analyze_job", "shortlist"]:
                step_bonus -= 0.2
                self.trust_score = max(0.5, self.trust_score - 0.1)
                feedback = "invalid step order"
            else:
                r_ids = getattr(action, "resumes", [])
                if not r_ids:
                    step_bonus -= 0.1 # Penalty for empty shortlist
                    self.trust_score = max(0.5, self.trust_score - 0.1)
                    feedback = "empty shortlist"
                else:
                    for r in r_ids:
                        if r not in self.shortlisted_resumes:
                            self.shortlisted_resumes.append(r)
                    step_bonus += 0.01
                    self.trust_score = min(1.0, self.trust_score + 0.05)
                    self.current_step_name = "rank"

        elif act_type == "rank":
            if self.current_step_name not in ["shortlist", "rank"]:
                step_bonus -= 0.2
                self.trust_score = max(0.5, self.trust_score - 0.1)
                feedback = "invalid step order"
            else:
                step_bonus += 0.01
                self.trust_score = min(1.0, self.trust_score + 0.05)
                self.current_step_name = "finalize"

        elif act_type == "finalize":
            if self.current_step_name not in ["rank", "finalize"]:
                step_bonus -= 0.2
                self.trust_score = max(0.5, self.trust_score - 0.1)
                feedback = "step skipping detected - penalized"
                done = True
            else:
                # Execution
                if self.task_type == "medium":
                    for job in self.jobs:
                        self.current_matches[job["id"]] = getattr(action, "ranked_list", [])
                else:
                    matches = getattr(action, "matches", {})
                    for j_id, r_id in matches.items():
                        self.current_matches[j_id] = r_id

                match_score = compute_reward(self.current_matches, self.gt, self.task_type)
                step_bonus += match_score
                
                if match_score > 0.5:
                    self.trust_score = min(1.0, self.trust_score + 0.05)
                else:
                    self.trust_score = max(0.5, self.trust_score - 0.1)
                
                done = True
        else:
            step_bonus -= 0.2
            self.trust_score = max(0.5, self.trust_score - 0.1)
            feedback = "invalid action_type"

        # -------------------------------------
        # CONSISTENCY & SCALE-AWARE ACCUMULATION
        # -------------------------------------
        # 1. Update internal raw total (including bonuses)
        old_raw_total = self.previous_step_score
        current_raw_total = old_raw_total + step_bonus
        
        self.previous_step_score = current_raw_total
        
        # 2. Apply Trust Scaling to the CUMULATIVE raw score
        scaled_cumulative = current_raw_total * self.trust_score
        
        # 3. Apply FINAL Soft-Clamp to the CUMULATIVE score (stay within 0-1 exclusively)
        # Maps [0, 1] -> [0.01, 0.99]
        final_cumulative = 0.01 + (0.98 * max(0.0, min(scaled_cumulative, 1.0)))
        
        # Return the FULL cumulative reward at this step
        # This makes the environment the single source of truth for the dashboard
        return self._build_observation(final_cumulative, done, feedback)
        
    def _finalize_step(self, done, reward, feedback):
        return self._build_observation(reward, done, feedback)

    def _build_observation(self, final_reward, done, feedback_msg):
        matched_str = []
        missing_str = []
        sugg = ""
        
        if self.current_matches:
            last_job_id = list(self.current_matches.keys())[-1]
            last_r_id = self.current_matches[last_job_id]
            if isinstance(last_r_id, list) and last_r_id:
                last_r_id = last_r_id[0] 
                
            job_obj = next((j for j in self.jobs if j["id"] == last_job_id), None)
            res_obj = next((r for r in self.resumes if r["id"] == last_r_id), None)
            
            if job_obj and res_obj:
                j_skills = [s.lower() for s in job_obj.get("skills_required", [])]
                r_skills = [s.lower() for s in res_obj.get("skills", [])]
                r_text = str(res_obj.get("text", "")).lower()
                
                matched_str = []
                for s in j_skills:
                    if s in r_skills or any(word in r_text for word in s.split('.')):
                        matched_str.append(s)
                
                missing_str = [s for s in j_skills if s not in matched_str]
                
                if missing_str:
                    sugg = f"Add {', '.join(missing_str)} to your profile"
                else:
                    sugg = "Perfect skill match!"

        return (
            Observation(
                resumes=self.resumes,
                jobs=self.jobs,
                current_matches=self.current_matches,
                step_count=self.step_count,
                shortlisted_resumes=self.shortlisted_resumes,
                rejected_resumes=self.rejected_resumes,
                current_step_name=self.current_step_name,
                confidence_score=self.confidence_score,
                trust_score=self.trust_score,
                action_history=self.action_history
            ),
            Reward(
                score=final_reward, 
                trust_score=self.trust_score,
                feedback=feedback_msg,
                matched_skills=matched_str,
                missing_skills=missing_str,
                suggestion=sugg
            ),
            done,
            {}
        )

    def state(self):
        return {
            "resumes": self.resumes,
            "jobs": self.jobs,
            "matches": self.current_matches,
            "steps": self.step_count,
            "task_type": self.task_type,
            "trust_score": self.trust_score
        }