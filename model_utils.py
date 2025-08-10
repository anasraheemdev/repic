import os
import re
import pickle
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
CSV_CANDIDATES = [
    os.path.join(DATA_DIR, "processed_reels_scripts.csv"),
    os.path.join(DATA_DIR, "reels_scripts.csv"),
    os.path.join(ROOT, "processed_reels_scripts.csv"),
    os.path.join(ROOT, "reels_scripts.csv"),
]

def _norm_face(x: str) -> str:
    x = (x or "").strip().lower()
    if x in {"y", "yes", "true", "1"}: return "yes"
    if x in {"n", "no", "false", "0"}: return "no"
    return "unknown"

def _clean(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = re.sub(r"http[s]?://\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _load_csv() -> pd.DataFrame:
    for p in CSV_CANDIDATES:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("CSV not found. Put it at data/processed_reels_scripts.csv or data/reels_scripts.csv")

    required = {"niche","type_of_content","show_face","script_text","hook_text"}
    cols = {c.lower() for c in df.columns}
    if required.issubset(cols):
        df = df.copy()
        df["niche"] = df["niche"].fillna("")
        df["type_of_content"] = df["type_of_content"].fillna("")
        df["show_face"] = df["show_face"].map(lambda v: _norm_face(str(v)))
        df["script_text"] = df["script_text"].fillna("")
        df["hook_text"] = df["hook_text"].fillna("")
        return df

    # Instagram-like mapping
    def get(*names):
        for n in names:
            for c in df.columns:
                if c.lower() == n:
                    return df[c]
        return pd.Series([""] * len(df))

    caption = get("caption").fillna("")
    first_comment = get("firstcomment").fillna("")
    niche = get("niche", "querytag", "ownerusername").fillna("general")
    toc = get("type_of_content").fillna("tips")
    face = get("show_face").fillna("unknown").map(lambda v: _norm_face(str(v)))

    def hook_from_text(t: str) -> str:
        t = t or ""
        parts = re.split(r"(?<=[.!?])\s+", t.strip())
        return (parts[0] if parts and parts[0] else t[:120]).strip()

    hooks = caption.apply(hook_from_text)
    scripts = caption.where(caption.str.len() > 0, first_comment).fillna("")

    out = pd.DataFrame({
        "niche": niche,
        "type_of_content": toc,
        "show_face": face,
        "script_text": scripts,
        "hook_text": hooks
    })
    out = out[(out["script_text"].str.strip()!="") | (out["hook_text"].str.strip()!="")].reset_index(drop=True)
    return out

def _combine_row(r: pd.Series) -> str:
    return (
        f"niche:{_clean(r['niche'])} | "
        f"type:{_clean(r['type_of_content'])} | "
        f"face:{_norm_face(r['show_face'])} | "
        f"script:{_clean(r['script_text'])} | "
        f"hook:{_clean(r['hook_text'])}"
    )

class ReelsRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2), dtype=np.float32, min_df=2)
        self.nn = NearestNeighbors(metric="cosine", n_neighbors=5)
        self.df = None
        self.X = None

    def train_or_load(self, model_path: str = MODEL_PATH) -> Tuple[bool, int]:
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
            self.vectorizer = obj["vectorizer"]
            self.nn = obj["nn"]
            self.df = obj["df"]
            self.X = obj["X"]
            return True, len(self.df)

        self.df = _load_csv()
        texts = self.df.apply(_combine_row, axis=1)
        self.X = self.vectorizer.fit_transform(texts)
        self.nn.fit(self.X)

        with open(model_path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "nn": self.nn, "df": self.df, "X": self.X}, f)

        return False, len(self.df)

    def recommend(self, niche: str, content_type: str, face: str, extra_text: str = "", k: int = 3) -> List[Dict]:
        # extra_text can include idea/audience/tone keywords to bias retrieval
        q = (
            f"niche:{_clean(niche)} | type:{_clean(content_type)} | face:{_norm_face(face)} | "
            f"{_clean(extra_text)} | script: | hook:"
        )
        qv = self.vectorizer.transform([q])
        dists, idxs = self.nn.kneighbors(qv, n_neighbors=min(k, self.df.shape[0]))
        out = []
        for rank, (i, dist) in enumerate(zip(idxs[0], dists[0]), start=1):
            row = self.df.iloc[i]
            out.append({
                "rank": rank,
                "distance": float(dist),
                "script_text": row["script_text"],
                "hook_text": row["hook_text"],
                "niche": row["niche"],
                "type_of_content": row["type_of_content"],
                "show_face": row["show_face"],
            })
        return out
