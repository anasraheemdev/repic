import os
import json
from dataclasses import dataclass
from typing import List, Optional

import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
VEC_DIR = os.path.join(ROOT, "vector_store")
os.makedirs(VEC_DIR, exist_ok=True)

INDEX_PATH = os.path.join(VEC_DIR, "index.faiss")
TEXTS_PATH = os.path.join(VEC_DIR, "texts.json")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast + small
CHUNK_WORDS = 220
CHUNK_OVERLAP = 40

PDF_CANDIDATES = [
    os.path.join(DATA_DIR, "viral_content.pdf"),
    os.path.join(ROOT, "viral_content.pdf"),
    os.path.join(DATA_DIR, "viral_content.txt"),
    os.path.join(ROOT, "viral_content.txt"),
]

def _find_pdf() -> Optional[str]:
    for p in PDF_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

def _read_pdf_or_txt(path: str) -> str:
    if path.endswith(".txt"):
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n".join(parts)

def _chunk_words(text: str, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+size]).strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        i = i + size - overlap
    return chunks

@dataclass
class RAGIndex:
    index: faiss.Index
    texts: List[str]
    model_name: str

class ViralRAG:
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []

    def build_or_load(self) -> RAGIndex:
        if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            self.texts = json.load(open(TEXTS_PATH, "r", encoding="utf-8"))
            return RAGIndex(self.index, self.texts, self.model_name)

        pdf_path = _find_pdf()
        if not pdf_path:
            raise FileNotFoundError("viral_content.pdf not found under /data or project root")

        raw = _read_pdf_or_txt(pdf_path)
        chunks = _chunk_words(raw)

        emb = self.model.encode(chunks, convert_to_numpy=True)
        faiss.normalize_L2(emb)                   # cosine via IP on normalized vectors
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)

        self.index = index
        self.texts = chunks

        faiss.write_index(index, INDEX_PATH)
        json.dump(chunks, open(TEXTS_PATH, "w", encoding="utf-8"))

        return RAGIndex(self.index, self.texts, self.model_name)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if self.index is None or not self.texts:
            return []
        q = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        return [self.texts[i] for i in I[0] if 0 <= i < len(self.texts)]
