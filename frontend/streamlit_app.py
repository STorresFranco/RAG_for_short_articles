import streamlit as st
from backend import rag  # directly import your RAG functions
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Initialize state (LLM + VectorDB)
if "llm" not in st.session_state or "vectordb" not in st.session_state:
    st.session_state.llm, st.session_state.vectordb = rag.initializer()

# ---------------- UI ----------------
st.title("RAG System for Short Articles")

# Tab structure
tab1, tab2 = st.tabs(["Populate DB", "Ask Questions"])

# -------- Populate DB tab --------
with tab1:
    st.header("Add URLs to Vector Database")
    urls_input = st.text_area("Enter URLs (one per line)")
    if st.button("Populate Database"):
        urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
        if urls:
            result = rag.populate_db(st.session_state.vectordb, urls)
            st.success(result["Status"])
        else:
            st.warning("Please enter at least one valid URL.")

# -------- Ask Questions tab --------
with tab2:
    st.header("Ask your question")
    query = st.text_input("Enter your query")
    if st.button("Get Answer"):
        if query.strip():
            try:
                answer = rag.qa_prediction(query, st.session_state.llm, st.session_state.vectordb)
                st.write("### Answer")
                st.json(answer)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question.")
