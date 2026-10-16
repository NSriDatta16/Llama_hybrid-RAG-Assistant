# ui/app.py
import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8008")

st.set_page_config(page_title="SmartGPT", page_icon="🧠", layout="wide")
st.title("🧠 SmartGPT")

with st.sidebar:
    st.markdown("### Settings")
    provider = st.selectbox("Provider", ["Gemini", "Groq", "OpenAI"], index=0)
    alpha = st.slider("Hybrid α (dense weight)", 0.0, 1.0, 0.6, 0.05)
    top_k = st.slider("Top-K", 1, 10, 5, 1)

    if provider == "Gemini":
        gemini_model = st.selectbox("Gemini model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
    elif provider == "Groq":
        groq_model = st.selectbox(
            "Groq model",
            ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
            index=0,
        )

    st.caption(f"Backend: [{API_BASE}]({API_BASE})")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, message in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(message)

q = st.chat_input("Ask anything about the knowledge base…")
if q:
    st.session_state.chat.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking…")
        if provider == "Gemini":
            endpoint = "/ask_gemini"
            model = gemini_model
        elif provider == "Groq":
            endpoint = "/ask_groq"
            model = groq_model
        else:
            endpoint = "/ask"
            model = None

        payload = {"query": q, "top_k": int(top_k), "alpha": float(alpha)}
        if model:
            payload["model"] = model

        try:
            response = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            answer = (data.get("answer") or "(no answer)").strip()
            placeholder.markdown(answer)
            st.session_state.chat.append(("assistant", answer))
        except Exception as exc:
            error_message = f"❌ **Error:** {exc}"
            placeholder.markdown(error_message)
            st.session_state.chat.append(("assistant", error_message))
