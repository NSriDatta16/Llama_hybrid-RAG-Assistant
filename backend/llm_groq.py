# backend/llm_groq.py
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

SUPPORTED_GROQ_MODELS = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
]
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")


def get_groq_llm(model_name: Optional[str] = None) -> ChatGroq:
    """Create a Groq client only when the Groq provider is selected."""
    api_key = (os.getenv("GROQ_API_KEY", "") or "").strip().strip('"').strip("'")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in backend/.env")
    model = (model_name or DEFAULT_GROQ_MODEL).strip()
    if model not in SUPPORTED_GROQ_MODELS:
        model = DEFAULT_GROQ_MODEL
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=0.3,
        max_tokens=1024,
    )


def rag_simple(query, retriever, llm, top_k=5):
    """Return a structured, tutorial-like answer using retrieved context."""
    docs = retriever.retrieve_query(query, top_k=top_k)
    context = "\n\n".join([d.page_content for d in docs]) if docs else ""
    if not context:
        return "I don't have enough reliable context to answer confidently."

    prompt = f"""
You are an expert data-science tutor.
Use ONLY the context below to craft a clear, educational explanation.
Structure the answer with a concise definition, key concepts, and a simple example.

Context:
{context}

Question: {query}

Answer:"""
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        return f"(Groq) generation failed: {exc}"
