# backend/main.py
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.models.schemas import AskRequest, AskResponse, RetrievedChunk
from backend.retrieval.embeddings import EmbeddingManager
from backend.retrieval.hybrid import HybridIndex

try:
    from llama_index.llms.openai import OpenAI
    from llama_index.core import PromptTemplate
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

from backend.llm_groq import get_groq_llm, rag_simple as groq_rag_simple

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

app = FastAPI(title="Hybrid RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this for production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_DIR = os.getenv("INDEX_DIR", "backend/index_wiki20k")
DEFAULT_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))
_EMB = None
_IDX = None


def ensure_loaded():
    """Load the embedder and hybrid index once, then reuse them."""
    global _EMB, _IDX
    if _EMB is None:
        _EMB = EmbeddingManager()
    if _IDX is None:
        _IDX = HybridIndex(base_dir=INDEX_DIR, dim=_EMB.dim)
        _IDX.load()
    return _EMB, _IDX


def retrieve(req: AskRequest):
    emb, idx = ensure_loaded()
    alpha = DEFAULT_ALPHA if req.alpha is None else req.alpha
    rows = idx.search(req.query, emb.encode, k=req.top_k, alpha=alpha)
    citations = [
        RetrievedChunk(doc_id=str(r["doc_id"]), score=float(r["score"]), text=r["text"])
        for r in rows
    ]
    context = "\n\n".join(r["text"] for r in rows)
    return alpha, citations, context, emb, idx


@app.get("/health")
def health():
    return {"status": "ok", "index_dir": INDEX_DIR}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """OpenAI-backed generation (optional)."""
    alpha, citations, context, _, _ = retrieve(req)
    if not _HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        return AskResponse(
            answer="OpenAI is not configured. Set OPENAI_API_KEY in backend/.env or choose Gemini/Groq.",
            citations=citations, used_alpha=alpha
        )
    llm = OpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    tmpl = PromptTemplate(
        "You are a precise RAG assistant. Use ONLY the context below. "
        "If the answer is not supported, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    answer = llm.complete(tmpl.format(context=context, query=req.query)).text
    return AskResponse(answer=answer, citations=citations, used_alpha=alpha)


class HybridRetriever:
    """Adapter for the existing Groq RAG helper."""
    def __init__(self, index, embedder):
        self.index = index
        self.embedder = embedder

    def retrieve_query(self, query: str, top_k: int = 5):
        rows = self.index.search(
            query, self.embedder.encode, k=top_k,
            alpha=float(os.getenv("HYBRID_ALPHA", DEFAULT_ALPHA))
        )
        class Doc:
            pass
        docs = []
        for row in rows:
            doc = Doc()
            doc.page_content = row["text"]
            doc.metadata = {"doc_id": row["doc_id"]}
            doc.score = row["score"]
            docs.append(doc)
        return docs


@app.post("/ask_groq", response_model=AskResponse)
def ask_groq(req: AskRequest):
    alpha, citations, _, emb, idx = retrieve(req)
    try:
        llm = get_groq_llm(req.model)
        answer = groq_rag_simple(req.query, HybridRetriever(idx, emb), llm, top_k=req.top_k)
        if not answer.strip():
            answer = "The Groq model returned an empty response."
    except Exception as exc:
        answer = f"Groq generation failed: {exc}"
    return AskResponse(answer=answer, citations=citations, used_alpha=alpha)


@app.post("/ask_gemini", response_model=AskResponse)
def ask_gemini(req: AskRequest):
    """Gemini-backed RAG using the Google Gen AI SDK."""
    alpha, citations, context, _, _ = retrieve(req)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return AskResponse(
            answer="Gemini is not configured. Add GEMINI_API_KEY to backend/.env and restart the API.",
            citations=citations, used_alpha=alpha
        )
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        model = req.model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        prompt = (
            "You are a precise Retrieval-Augmented Generation assistant. Answer using ONLY "
            "the supplied context. If the context does not support an answer, say you do not know. "
            "Be clear and concise.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION:\n{req.query}\n\nANSWER:"
        )
        response = client.models.generate_content(model=model, contents=prompt)
        answer = (response.text or "").strip() or "Gemini returned an empty response."
    except Exception as exc:
        answer = f"Gemini generation failed: {exc}"
    return AskResponse(answer=answer, citations=citations, used_alpha=alpha)
