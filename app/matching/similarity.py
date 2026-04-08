import os
import warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from tqdm import tqdm
tqdm.disable = True
# 🔥 suppress all unwanted logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
warnings.filterwarnings("ignore")

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np


# 🔥 global model (lazy load)
model = None


def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        model.max_seq_length = 256
    return model


def compute_similarity(resumes, jobs):
    # ---------- TEXT PREP ----------
    r_text = [str(r["text"]).lower() for r in resumes]

    j_text = [
        (j["description"] + " " + " ".join(j.get("skills_required", []) * 3)).lower()
        for j in jobs
    ]

    # ---------- TF-IDF ----------
    corpus = r_text + j_text
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(corpus)

    r_tfidf = tfidf[:len(r_text)]
    j_tfidf = tfidf[len(r_text):]

    tfidf_sim = cosine_similarity(j_tfidf, r_tfidf)

    # ---------- TRANSFORMER ----------
    model = get_model()

    r_emb = model.encode(r_text, show_progress_bar=False)
    j_emb = model.encode(j_text, show_progress_bar=False)

    emb_sim = cosine_similarity(j_emb, r_emb)

    # ---------- FINAL SCORE ----------
    final_sim = 0.9 * tfidf_sim + 0.1 * emb_sim

    return final_sim