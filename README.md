



# Llama-Hybrid-RAG-Assistant  

### *An AI Retrieval-Augmented Generation Assistant built with Llama 3, FAISS, BM25, and persistent memory.*
---
## 🚀 Overview
The **Llama Hybrid RAG Assistant** is a **Retrieval-Augmented Generation (RAG)** system powered by **Groq’s Llama 3**, **FAISS dense embeddings**, and **BM25 sparse retrieval**.  
It enables intelligent, context-aware conversations over large text datasets (Wikipedia, StackExchange, arXiv, etc.) using **hybrid retrieval** and **persistent chat memory**.

This project demonstrates a complete **local RAG pipeline** with:
- FastAPI backend  
- Streamlit chat frontend  
- Hybrid search (FAISS + BM25)  
- Llama-3 (Groq API) inference  
- Conversational memory persistence  

---

## 🧩 Architecture

```plaintext
                ┌───────────────────────────────┐
                │           User (UI)           │
                │   Streamlit Chat Interface     │
                └───────────────┬───────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │      FastAPI Backend    │
                    │   (backend/main.py)     │
                    ├────────────────────────┤
                    │  Retrieval Layer        │
                    │   • FAISS (dense)       │
                    │   • BM25 (sparse)       │
                    │   • Hybrid Combiner     │
                    ├────────────────────────┤
                    │  LLM Inference Layer    │
                    │   • Groq (Llama 3 API)  │
                    │   • OpenAI (fallback)   │
                    ├────────────────────────┤
                    │  Memory + Session Store │
                    │   • JSON persistence    │
                    └────────────────────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │      Local Data Sources     │
                  │  Wikipedia / StackExchange  │
                  │  arXiv / Custom Text Files  │
                  └────────────────────────────┘
