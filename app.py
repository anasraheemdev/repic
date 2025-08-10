import os
import json
import argparse
from dotenv import load_dotenv

from flask import Flask, render_template, request, make_response
from model_utils import ReelsRecommender
from rag_utils import ViralRAG
import requests

load_dotenv()
app = Flask(__name__, template_folder="templates")

# ---------- Globals
recommender = ReelsRecommender()
rag = ViralRAG()
MODEL_READY = False
RAG_READY = False


# ---------- Groq LLM rewrite (RAG stays internal)
def groq_rewrite(niche: str,
                 content_type: str,
                 face: str,
                 basic_idea: str,
                 audience: str,
                 tone: str,
                 base_hook: str,
                 base_script: str,
                 rag_points: list):
    """Return (used, hook, script). Uses Groq if key available, otherwise falls back to dataset output."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return False, base_hook, base_script

    system = (
        "You are a professional short-form video copywriter. "
        "Deliver a high-retention HOOK (≤ 20 words) and a SCRIPT (60–100 words) for Instagram Reels/TikTok. "
        "Be crisp, concrete, and actionable. Keep to the requested tone. No hashtags or emojis."
    )

    # RAG stays in logic—summarized bullets only, never displayed
    rag_bullets = "\n".join(f"- {p}" for p in rag_points[:3])

    user = f"""
Niche: {niche}
Type of content: {content_type}
Show face on camera: {face}
Basic idea: {basic_idea}
Audience persona: {audience}
Tone/style: {tone}

Seed suggestion from dataset:
HOOK: {base_hook}
SCRIPT: {base_script}

Relevant strategy guidance (internal only):
{rag_bullets}

Task: Rewrite the HOOK and SCRIPT to maximize watch-time and saves for this niche and content type.
Return compact JSON: {{"hook": "...", "script": "..."}} only.
""".strip()

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.6,
        "max_tokens": 350,
    }

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        # Extract JSON
        import re as _re
        m = _re.search(r"\{.*\}", content, _re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        hook = (data.get("hook") or base_hook).strip()
        script = (data.get("script") or base_script).strip()
        return True, hook, script
    except Exception as e:
        print("[Groq] Warning:", e)
        return False, base_hook, base_script


# ---------- Bootstrap/load
def ensure_ready(force_rebuild=False):
    global MODEL_READY, RAG_READY
    if force_rebuild:
        try: os.remove(os.path.join("models", "model.pkl"))
        except FileNotFoundError: pass
        for p in ["vector_store/index.faiss", "vector_store/texts.json"]:
            try: os.remove(p)
            except FileNotFoundError: pass

    if not MODEL_READY or force_rebuild:
        loaded, n = recommender.train_or_load()
        MODEL_READY = True
        print(f"[Model] {'Loaded' if loaded else 'Trained'} with {n} rows.")

    if not RAG_READY or force_rebuild:
        try:
            rag.build_or_load()
            RAG_READY = True
            print("[RAG] Ready.")
        except Exception as e:
            print("[RAG] Warning:", e)
            RAG_READY = False


# ---------- Routes
@app.route("/", methods=["GET"])
def home():
    ensure_ready(False)
    return render_template("index.html", result=None, groq_set=bool(os.getenv("GROQ_API_KEY")))

@app.route("/rebuild", methods=["POST"])
def rebuild():
    ensure_ready(True)
    return render_template("index.html", result=None, groq_set=bool(os.getenv("GROQ_API_KEY")))

@app.route("/recommend", methods=["POST"])
def recommend_route():
    ensure_ready(False)

    # 6 inputs
    niche = request.form.get("niche", "").strip()
    toc = request.form.get("type_of_content", "").strip()
    basic_idea = request.form.get("basic_idea", "").strip()
    audience = request.form.get("audience_persona", "").strip()
    tone = request.form.get("tone_style", "").strip()
    face = request.form.get("show_face", "unknown").strip()

    # Bias TF-IDF with extra context
    extra_text = f"{basic_idea} {audience} {tone}"
    recs = recommender.recommend(niche, toc, face, extra_text=extra_text, k=3)
    if not recs:
        return render_template("index.html", result={"error": "No suitable examples found. Add more rows to your CSV."}, groq_set=bool(os.getenv("GROQ_API_KEY")))

    best = recs[0]
    alternates = [r["hook_text"] for r in recs[1:]]

    # RAG (internal only: used to guide Groq, never shown)
    rag_query = f"{niche} {toc} idea={basic_idea} audience={audience} tone={tone} show_face={face} viral video strategies"
    rag_points = rag.retrieve(rag_query, k=3) if RAG_READY else []

    # Groq rewrite
    used, final_hook, final_script = groq_rewrite(
        niche=niche, content_type=toc, face=face,
        basic_idea=basic_idea, audience=audience, tone=tone,
        base_hook=best["hook_text"], base_script=best["script_text"],
        rag_points=rag_points
    )

    # Build result (only clean, user-facing text)
    result = {
        "inputs": {
            "niche": niche,
            "type_of_content": toc,
            "basic_idea": basic_idea,
            "audience": audience,
            "tone": tone,
            "show_face": face,
        },
        "final_hook": final_hook,
        "final_script": final_script,
        "groq_used": used,
        "seed_hook": best["hook_text"],
        "seed_script": best["script_text"],
        "alternates": alternates
    }
    return render_template("index.html", result=result, groq_set=bool(os.getenv("GROQ_API_KEY")))

@app.route("/download", methods=["POST"])
def download():
    hook = (request.form.get("hook") or "").strip()
    script = (request.form.get("script") or "").strip()
    text = f"HOOK:\n{hook}\n\nSCRIPT:\n{script}\n"
    resp = make_response(text)
    resp.headers["Content-Type"] = "text/plain; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=viral_reel.txt"
    return resp


# ---------- CLI fallback
def run_cli(args):
    ensure_ready(False)
    niche = args.niche or input("Niche: ").strip()
    toc = args.type or input("Type of content: ").strip()
    basic_idea = input("Basic idea: ").strip()
    audience = input("Audience persona: ").strip()
    tone = input("Tone/style: ").strip()
    face = args.face or input("Show face? (yes/no/unknown): ").strip()

    extra_text = f"{basic_idea} {audience} {tone}"
    recs = recommender.recommend(niche, toc, face, extra_text=extra_text, k=3)
    if not recs:
        print("No recommendations found. Add more rows to your CSV.")
        return
    best = recs[0]
    rag_points = rag.retrieve(f"{niche} {toc} {extra_text} show_face={face} viral strategy", k=3) if RAG_READY else []
    used, final_hook, final_script = groq_rewrite(niche, toc, face, basic_idea, audience, tone,
                                                 best["hook_text"], best["script_text"], rag_points)
    print("\n=== Final Output ===")
    print("Hook:", final_hook)
    print("Script:", final_script)
    if not used:
        print("\n(LLM rewrite skipped: set GROQ_API_KEY to enable.)")
    print("\n— Seed (from dataset) —")
    print("Hook:", best["hook_text"])
    print("Script:", best["script_text"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Viral Reels App (Flask + CLI)")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of Flask")
    parser.add_argument("--niche", type=str, default=None)
    parser.add_argument("--type", type=str, default=None)
    parser.add_argument("--face", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.cli:
        run_cli(args)
    else:
        ensure_ready(False)
        app.run(host=args.host, port=args.port, debug=True)