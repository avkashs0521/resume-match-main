import re


def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text


def skill_in_text(skill, text):
    skill_words = skill.split()
    return any(word in text for word in skill_words)


def generate_feedback(resume, job):
    resume_text = normalize(resume.get("text", ""))
    job_skills = job.get("skills_required", [])

    matched = []
    missing = []

    for skill in job_skills:
        skill_clean = normalize(skill)

        if skill_in_text(skill_clean, resume_text):
            matched.append(skill)
        else:
            missing.append(skill)

    score = len(matched) / (len(job_skills) + 1)

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "score": round(score, 2),
        "suggestion": f"Add skills like {', '.join(missing[:3])}" if missing else "Great fit!"
    }