def compute_reward(pred, ground_truth, task_type):
    if not pred or not ground_truth:
        return 0.0

    if task_type == "easy":
        job_id = list(ground_truth.keys())[0]
        correct = pred.get(job_id) == ground_truth[job_id]
        return 1.0 if correct else 0.0

    elif task_type == "medium":
        job_id = list(ground_truth.keys())[0]
        gt = ground_truth.get(job_id, [])
        pred_list = pred.get(job_id, [])

        score = 0.0
        for i, r in enumerate(pred_list):
            if r in gt:
                score += 1.0 / (i + 1)

        return min(1.0, score / len(gt)) if gt else 0.0

    elif task_type == "hard":
        correct = 0
        total = len(ground_truth)
        for job, correct_resume in ground_truth.items():
            if pred.get(job) == correct_resume:
                correct += 1

        return correct / total if total > 0 else 0.0
    
    return 0.0