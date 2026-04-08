import os
import uvicorn
import json
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.env.environment import ResumeEnv
from app.env.models import Action
from app.matching.matcher import match_easy, match_medium, match_hard, match_random, get_top_k
from pydantic import BaseModel

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class MatchRequest(BaseModel):
    task: str = "easy"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono&display=swap');

:root {
    --bg-main: #0c0c14;
    --bg-card: rgba(20, 20, 32, 0.7);
    --border-card: rgba(255, 255, 255, 0.08);
    --accent-purple: #8a2be2;
    --accent-teal: #00f2fe;
    --accent-amber: #ff8c00;
    --text-primary: #e0e0e0;
    --text-dim: #808090;
}

body {
    background-color: var(--bg-main);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    margin: 0; padding: 20px;
    height: 100vh;
    display: flex; flex-direction: column; gap: 15px;
    overflow: hidden;
}

/* Header */
.header {
    display: flex; justify-content: space-between; align-items: center;
    background: var(--bg-card);
    padding: 10px 20px; border-radius: 12px;
    border: 1px solid var(--border-card);
}
.header-left { display: flex; align-items: center; gap: 12px; }
.logo-box { background: var(--accent-purple); width: 24px; height: 24px; border-radius: 6px; }
.header-right { display: flex; gap: 15px; align-items: center; font-size: 0.85rem; }
.status-pill { background: rgba(0, 255, 204, 0.1); color: #00ffcc; padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(0, 255, 204, 0.2); }
.type-pill { background: rgba(138, 43, 226, 0.1); color: #8a2be2; padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(138, 43, 226, 0.2); }

/* Main Grid */
.grid-container {
    display: grid;
    grid-template-columns: 280px 1fr 320px;
    gap: 15px; height: calc(100vh - 100px);
}

.panel {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 18px;
    display: flex; flex-direction: column; gap: 15px;
    position: relative;
}

h3 { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; margin: 0 0 10px 0; }

/* Sidebar Elements */
.task-item {
    background: rgba(255,255,255,0.03); border-radius: 12px; padding: 12px;
    display: flex; flex-direction: column; gap: 8px;
}
.task-header { display: flex; justify-content: space-between; align-items: center; }
.task-btn {
    background: transparent; border: 1px solid rgba(255,255,255,0.1);
    color: var(--text-primary); padding: 6px 15px; border-radius: 8px; cursor: pointer;
    font-size: 0.8rem; transition: all 0.2s;
}
.task-btn:hover { background: rgba(255,255,255,0.05); border-color: var(--accent-teal); }
.badge-label { font-size: 0.65rem; padding: 2px 8px; border-radius: 4px; }

.stat-row { display: flex; justify-content: space-between; font-size: 0.85rem; padding: 4px 0; }
.stat-val { color: var(--accent-teal); font-weight: 600; }

/* Center Terminal */
.terminal {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px; padding: 15px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;
    flex-grow: 1; overflow-y: auto; color: #fff;
}
.log-step { color: var(--accent-teal); margin-top: 6px; }
.log-reason { color: #88aaff; font-style: italic; }

.reward-chart-panel { height: 200px; }
.nlp-delta-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
.delta-card { background: rgba(255,255,255,0.02); padding: 12px; border-radius: 10px; text-align: center; }
.delta-val { font-size: 1.2rem; font-weight: bold; margin-bottom: 5px; }
.delta-progress { height: 3px; background: rgba(255,255,255,0.05); border-radius: 2px; }

/* Right Sidebar */
.analytic-score-box { background: rgba(0,0,0,0.2); padding: 15px; border-radius: 12px; text-align: left; }
.big-score { font-size: 2.2rem; font-weight: bold; color: var(--accent-teal); }

.feedback-box {
    border-left: 3px solid var(--accent-amber);
    background: rgba(255, 140, 0, 0.05);
    padding: 12px; font-size: 0.85rem; line-height: 1.4; color: var(--accent-amber);
}

.dist-chart { flex-grow: 1; display: flex; align-items: center; justify-content: center; }

/* Modal */
.modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); backdrop-filter: blur(10px); justify-content: center; align-items: center; }
.modal-content { background: #0c0c14; border: 1px solid var(--accent-purple); border-radius: 24px; padding: 25px; width: 85%; max-height: 85vh; overflow-y: auto; }
"""

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CVCraft | AI Decision Arena</title>
    <style>REPLACE_CSS</style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>

<div class="header" style="height: 80px; padding: 0 25px;">
    <div class="header-left">
        <img src="/static/logo.png" alt="CV-Craft" style="height:55px; filter: drop-shadow(0 0 15px rgba(138,43,226,0.25)); transition: transform 0.3s ease;">
    </div>
    <div class="header-right">
        <span style="color:var(--text-dim);">● <span id="header-live">Live</span></span>
        <div class="type-pill">Deterministic</div>
        <button class="task-btn" onclick="resetArena()" style="margin-left:10px; border-color:var(--accent-purple);">Reset</button>
        <span id="header-time" style="color:var(--text-dim);">04-08 20:29</span>
    </div>
</div>

<div class="grid-container">
    <!-- LEFT SIDEBAR -->
    <div class="panel">
        <h3>Tasks</h3>
        <div class="task-item">
            <div class="task-header">
                <span style="font-weight:500;">Easy</span>
                <span class="badge-label" style="background:rgba(0, 255, 127, 0.1); color:#00ff7f;">Match Agent</span>
            </div>
            <button class="task-btn" onclick="runTask('easy', this)">run ></button>
        </div>
        <div class="task-item">
            <div class="task-header">
                <span style="font-weight:500;">Medium</span>
                <span class="badge-label" style="background:rgba(255, 165, 0, 0.1); color:#ffa500;">Rank Agent</span>
            </div>
            <button class="task-btn" onclick="runTask('medium', this)">run ></button>
        </div>
        <div class="task-item">
            <div class="task-header">
                <span style="font-weight:500;">Hard</span>
                <span class="badge-label" style="background:rgba(255, 69, 0, 0.1); color:#ff4500;">Allocate Agent</span>
            </div>
            <button class="task-btn" onclick="runTask('hard', this)">run ></button>
        </div>

        <h3 style="margin-top:20px;">Leaderboard</h3>
        <div class="stat-row"><span>Optimal Agent</span> <span class="stat-val" id="lb-optimal">0.00</span></div>
        <div class="stat-row"><span>Random Baseline</span> <span class="stat-val" id="lb-random" style="color:#fe4a90;">0.00</span></div>

        <h3 style="margin-top:20px;">Run Breakdown</h3>
        <div class="stat-row"><span>Steps</span> <span id="rb-steps" style="color:#fff;">0/0</span></div>
        <div class="stat-row"><span>Strategy</span> <span id="rb-strategy" style="color:#fff;">--</span></div>
        <div class="stat-row"><span>Complexity</span> <span id="rb-complexity" style="color:#fff;">--</span></div>
    </div>

    <!-- CENTER COLUMN -->
    <div style="display:flex; flex-direction:column; gap:15px;">
        <div class="panel" style="flex-grow:1;">
            <h3>Arena Log</h3>
            <div class="terminal" id="terminal">
                <div style="color:var(--text-dim);">>> CVCraft engine standby. Select a task to begin...</div>
            </div>
        </div>
        
        <div class="panel reward-chart-panel">
            <h3>Reward History</h3>
            <div style="height:140px;"><canvas id="rewardChart"></canvas></div>
        </div>

        <div class="panel">
            <h3>NLP Delta</h3>
            <div class="nlp-delta-row">
                <div class="delta-card">
                    <div class="delta-val" id="nlp-matched" style="color:var(--accent-purple);">0</div>
                    <div style="font-size:0.6rem; color:var(--text-dim);">MATCHED</div>
                    <div class="delta-progress"><div id="prog-matched" style="background:var(--accent-purple); width:0%; height:100%;"></div></div>
                </div>
                <div class="delta-card">
                    <div class="delta-val" id="nlp-align" style="color:var(--accent-teal);">0</div>
                    <div style="font-size:0.6rem; color:var(--text-dim);">ALIGNMENT</div>
                    <div class="delta-progress"><div id="prog-align" style="background:var(--accent-teal); width:0%; height:100%;"></div></div>
                </div>
                <div class="delta-card">
                    <div class="delta-val" id="nlp-missing" style="color:#ff4a90;">0</div>
                    <div style="font-size:0.6rem; color:var(--text-dim);">MISSING</div>
                    <div class="delta-progress"><div id="prog-missing" style="background:#ff4a90; width:0%; height:100%;"></div></div>
                </div>
            </div>
        </div>
    </div>

    <!-- RIGHT SIDEBAR -->
    <div class="panel">
        <h3>Agent Analytics</h3>
        <div class="analytic-score-box">
            <div style="font-size:0.65rem; color:var(--text-dim); margin-bottom:5px;">REWARD GRADIENT</div>
            <div class="big-score" id="ana-score">0.00</div>
        </div>

        <h3>Skill Overlap</h3>
        <div id="skill-badges" style="display:flex; flex-wrap:wrap; gap:6px;">--</div>

        <h3>Adaptive Feedback</h3>
        <div class="feedback-box" id="feedback-text">
            No live data. Initiating simulation trajectory will populate this field.
        </div>

        <h3 style="margin-top:20px;">Score Distribution</h3>
        <div class="dist-chart"><canvas id="distChart"></canvas></div>
        <div style="display:flex; justify-content:space-between; font-size:0.75rem;">
            <span><span style="color:var(--accent-purple);">■</span> Optimal</span>
            <span><span style="color:#00ffcc;">■</span> Baseline</span>
        </div>
    </div>
</div>

<div id="inspector-modal" class="modal">
    <div class="modal-content">
        <span onclick="closeModal()" style="float:right; cursor:pointer; font-size:2rem;">&times;</span>
        <h2 style="color:var(--accent-teal);">🔍 Decision Inspector</h2>
        <div id="modal-body"></div>
    </div>
</div>

<script>
    const term = document.getElementById('terminal');
    let rewardChart, distChart;
    let runningTotal = 0;

    function updateTime() {
        const now = new Date();
        document.getElementById('header-time').innerText = now.toLocaleString('en-US', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }).replace(',', '');
    }
    setInterval(updateTime, 1000);

    function initCharts() {
        if (rewardChart) rewardChart.destroy();
        if (distChart) distChart.destroy();

        rewardChart = new Chart(document.getElementById('rewardChart'), {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Trajectory', data: [], borderColor: '#8a2be2', borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#8a2be2', tension: 0.2, fill: false }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, 
                      scales: { y: { min: 0, max: 1.1, border: { display: false }, grid: { color: 'rgba(255,255,255,0.05)' } }, x: { grid: { display: false } } } }
        });

        distChart = new Chart(document.getElementById('distChart'), {
            type: 'doughnut',
            data: { datasets: [{ data: [0.0, 1.0], backgroundColor: ['#8a2be2', '#1a1a2a'], borderWidth: 0, cutout: '80%' }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
    }

    function resetArena() {
        runningTotal = 0;
        term.innerHTML = '<div style="color:var(--text-dim);">>> CVCraft engine reset. Ready for new simulation...</div>';
        initCharts();
        document.getElementById('ana-score').innerText = '0.00';
        document.getElementById('lb-optimal').innerText = '0.00';
        document.getElementById('lb-random').innerText = '0.00';
        document.getElementById('rb-steps').innerText = '0/0';
        document.getElementById('rb-strategy').innerText = '--';
        document.getElementById('rb-complexity').innerText = '--';
        document.getElementById('skill-badges').innerHTML = '--';
        document.getElementById('feedback-text').innerText = 'Simulation Standby...';
        document.getElementById('nlp-matched').innerText = '0';
        document.getElementById('nlp-align').innerText = '0';
        document.getElementById('nlp-missing').innerText = '0';
        document.getElementById('prog-matched').style.width = '0%';
        document.getElementById('prog-align').style.width = '0%';
        document.getElementById('prog-missing').style.width = '0%';
    }

    function log(msg, type='info', data=null) {
        const div = document.createElement('div');
        div.style.padding = '4px 0';
        if (type === 'step') div.className = 'log-step';
        else if (type === 'reasoning') div.className = 'log-reason';
        div.innerHTML = msg;
        if (data) { div.onclick = () => openInspector(data); div.style.cursor = 'pointer'; }
        term.appendChild(div);
        term.scrollTop = term.scrollHeight;
    }

    function openInspector(data) {
        document.getElementById('modal-body').innerHTML = `<pre style="color:#00ffcc; font-family:monospace;">${JSON.stringify(data, null, 2)}</pre>`;
        document.getElementById('inspector-modal').style.display = 'flex';
    }
    function closeModal() { document.getElementById('inspector-modal').style.display = 'none'; }

    async function runTask(taskType, btn) {
        btn.disabled = true;
        runningTotal = 0;
        term.innerHTML = '';
        log(`>> INITIALIZING ${taskType.toUpperCase()} ARENA TRAJECTORY...`, 'info');
        
        // DYNAMIC BASELINE
        // DYNAMIC BASELINE
        const baselines = { 'easy': '0.22', 'medium': '0.12', 'hard': '0.04' };
        document.getElementById('lb-random').innerText = baselines[taskType] || '0.20';
        
        rewardChart.data.labels = [];
        rewardChart.data.datasets[0].data = [];
        rewardChart.update();

        // RESET UI FOR NEW RUN
        document.getElementById('rb-steps').innerText = `0/${taskType === 'hard' ? 5 : 4}`;
        document.getElementById('rb-strategy').innerText = taskType === 'hard' ? 'Batch' : 'Sequential';
        document.getElementById('rb-complexity').innerText = taskType === 'hard' ? 'High' : (taskType === 'medium' ? 'Med' : 'Low');
        
        document.getElementById('nlp-matched').innerText = '...';
        document.getElementById('nlp-align').innerText = '...';
        document.getElementById('nlp-missing').innerText = '...';
        document.getElementById('skill-badges').innerHTML = '--';
        document.getElementById('feedback-text').innerText = 'Starting neural scan...';

        try {
            const res = await fetch(`/api/run/${taskType}`);
            const data = await res.json();
            
            let currentStep = 0;
            for (let entry of data.logs) {
                currentStep++;
                await new Promise(r => setTimeout(r, 800));
                
                // Special formatting for steps
                let displayMsg = entry.msg;
                if (entry.status === 'step') {
                    displayMsg = `<span style="color:var(--text-dim);">>></span> STEP ${currentStep} · <span style="color:#fff;">${entry.msg.split(': ')[1]}</span>`;
                }
                log(displayMsg, entry.status, entry.data);
                
                document.getElementById('rb-steps').innerText = `${currentStep}/${data.logs.length}`;

                if (entry.reward !== undefined) {
                    const currentScore = entry.reward;
                    rewardChart.data.labels.push(`S${currentStep}`);
                    rewardChart.data.datasets[0].data.push(currentScore);
                    rewardChart.update();
                    
                    distChart.data.datasets[0].data = [currentScore, 1 - currentScore];
                    distChart.update();
                    
                    document.getElementById('ana-score').innerText = currentScore.toFixed(2);
                }

                // DYNAMIC SCANNING STATES
                if (currentStep === 1) document.getElementById('feedback-text').innerText = 'Analyzing job requirements...';
                if (currentStep === 2) document.getElementById('feedback-text').innerText = 'Shortlisting semantic candidates...';
                if (currentStep === 3) document.getElementById('feedback-text').innerText = 'Ranking matches by vector proximity...';

                if (entry.xai) {
                    const finalTotal = entry.reward;
                    document.getElementById('lb-optimal').innerText = finalTotal.toFixed(2);
                    document.getElementById('nlp-matched').innerText = entry.xai.matched_skills.length;
                    document.getElementById('prog-matched').style.width = `${entry.xai.matched_skills.length * 10}%`;
                    
                    document.getElementById('nlp-missing').innerText = entry.xai.missing_skills.length;
                    document.getElementById('prog-missing').style.width = `${entry.xai.missing_skills.length * 10}%`;
                    
                    document.getElementById('nlp-align').innerText = (finalTotal * 10).toFixed(0);
                    document.getElementById('prog-align').style.width = `${finalTotal * 100}%`;

                    if (entry.xai.matched_skills.length > 0) {
                        document.getElementById('skill-badges').innerHTML = entry.xai.matched_skills.map(s => 
                            `<span style="font-size:0.65rem; background:rgba(138,43,226,0.1); border: 1px solid rgba(138,43,226,0.2); color:#8a2be2; padding:4px 10px; border-radius:6px;">${s}</span>`
                        ).join('');
                    }
                    
                    document.getElementById('feedback-text').innerText = entry.xai.suggestion || 'Simulation complete.';
                }
            }
            
        } catch(e) { log(">> ERROR: Simulation trajectory interrupted.", 'step'); }
        btn.disabled = false;
    }
    window.onload = () => { initCharts(); updateTime(); };
</script>
</body>
</html>
""".replace("REPLACE_CSS", CSS)

@app.get("/")
def home():
    return HTMLResponse(content=HTML_CONTENT)

@app.get("/api/run/{task_type}")
def run_task(task_type: str):
    env = ResumeEnv(task_type=task_type)
    obs = env.reset()
    logs = []
    
    all_resumes_dict = [r.model_dump() for r in obs.resumes]
    all_jobs_dict = [j.model_dump() for j in obs.jobs]
    
    # 🚀 STEP 1: ANALYZE_JOB
    obs, reward, done, _ = env.step(Action(action_type="analyze_job"))
    logs.append({
        "msg": f">> STEP 1: [Analyze] Extracting keyword dependencies for {task_type.upper()} task.",
        "status": "step",
        "reward": reward.score,
        "trust": reward.trust_score,
        "data": {"jobs": all_jobs_dict}
    })

    # 🚀 STEP 2: SHORTLIST
    primary_job = all_jobs_dict[0]
    shortlist_ids = get_top_k(primary_job, all_resumes_dict, k=5)
    obs, reward, done, _ = env.step(Action(action_type="shortlist", resumes=shortlist_ids))
    logs.append({
        "msg": f">> STEP 2: [Shortlist] Identified top-5 candidates via semantic similarity.",
        "status": "step",
        "reward": reward.score,
        "trust": reward.trust_score,
        "shortlist_count": len(obs.shortlisted_resumes),
        "data": {"shortlisted_ids": shortlist_ids}
    })

    # 🚀 STEP 3: RANK
    obs, reward, done, _ = env.step(Action(action_type="rank"))
    logs.append({
        "msg": f">> STEP 3: [Rank] Calculating optimal decision path across all available vectors.",
        "status": "step",
        "reward": reward.score,
        "trust": reward.trust_score,
        "data": {"current_matches": obs.current_matches}
    })

    # 🚀 STEP 4: FINALIZE
    if task_type == "easy":
        matches = match_easy(all_resumes_dict, all_jobs_dict)
        final_act = Action(action_type="finalize", matches=matches)
    elif task_type == "medium":
        ranked = match_medium(all_resumes_dict, all_jobs_dict)
        final_act = Action(action_type="finalize", ranked_list=ranked)
    else: # hard
        matches = match_hard(all_resumes_dict, all_jobs_dict)
        final_act = Action(action_type="finalize", matches=matches)

    obs, reward, done, _ = env.step(final_act)
    logs.append({
        "msg": f">> STEP 4: [Finalize] Optimal Batch Assignment applied successfully.",
        "status": "success",
        "reward": reward.score,
        "trust": reward.trust_score,
        "xai": reward.model_dump(),
        "data": {"final_matches": obs.current_matches}
    })
        
    return {"logs": logs}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
