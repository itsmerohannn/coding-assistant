import streamlit as st  # type: ignore[import]

from src.assistant import run_agent
from src.models import load_generation_pipeline
from src.rag import CodeChunk, CodeIndexer

st.set_page_config(page_title="AI Coding Assistant Agent", layout="wide")
st.title("🤖 AI Coding Assistant Agent")
st.caption("Python • Hugging Face • FAISS • RAG • Streamlit")

with st.sidebar:
    st.header("Settings")
    task = st.selectbox("Task", ["Explain", "Review", "Improve"])
    st.subheader("Knowledge Base")
    kb_text = st.text_area(
        "Paste reference code/documentation snippets (one per section, separated by \n---\n)",
        height=200,
    )

if "generator" not in st.session_state:
    with st.spinner("Loading generation model..."):
        st.session_state.generator = load_generation_pipeline()

if "indexer" not in st.session_state:
    st.session_state.indexer = CodeIndexer()

if kb_text:
    chunks = [
        CodeChunk(text=chunk.strip(), source=f"snippet_{i+1}")
        for i, chunk in enumerate(kb_text.split("\n---\n"))
        if chunk.strip()  # Fixed the syntax error here
    ]
    if chunks:
        st.session_state.indexer.build(chunks)

col1, col2 = st.columns(2)
with col1:
    code_input = st.text_area("Python code", height=320, placeholder="Paste Python code here...")
with col2:
    query = st.text_area("Question / goal", height=160, placeholder="What do you want help with?")

if st.button("Run Assistant", type="primary"):
    if not code_input.strip():
        st.warning("Please provide Python code.")
    else:
        with st.spinner("Thinking..."):
            response = run_agent(
                st.session_state.generator,
                st.session_state.indexer,
                task=task,
                code=code_input,
                user_query=query,
        )
        st.subheader("Assistant Response")
        st.markdown(response)
