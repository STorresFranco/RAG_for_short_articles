import streamlit as st
import sys, os

# Get absolute path to the repo root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Now imports should work
from backend import rag

import streamlit as st
from PIL import Image
from backend import rag  # import your RAG functions

# ---------------- Sidebar ---------------- #
with st.sidebar:
    st.header("Provide URLs")
    url1 = st.text_input("URL 1")
    url2 = st.text_input("URL 2")
    url3 = st.text_input("URL 3")

    if st.button("Process URLs"):
        urls = [url for url in [url1, url2, url3] if url.strip()]
        if not urls:
            st.error("⚠️ You must provide at least 1 valid URL.")
        else:
            if "llm" not in st.session_state or "vectordb" not in st.session_state:
                # Initialize system once before populating
                st.session_state.llm, st.session_state.vectordb = rag.initializer()

            with st.spinner("Processing URLs... Please wait."):
                try:
                    result = rag.populate_db(st.session_state.vectordb, urls)
                    status_msg = result.get("Status", "No status returned")
                    st.success(f"✅ Successfully processed the URLs | {status_msg}")
                except Exception as e:
                    st.error(f"❌ Failed to process URLs: {str(e)}")

# ---------------- Main Page ---------------- #
st.title("🔎 RAG Project Implementation")

st.markdown("""
### Hi!  
In this project I've implemented a basic **RAG system**.  
It takes information from up to three URLs you provide and resolves prompts about them! 🚀
            
#### Recomendations
* Please, be patient with the url processing, this app is based on free tools so it might take a little.
* I recommend using this app to analyze short articles or news, that way It will take less time processing the page content
""")

tab1, tab2 = st.tabs(["ℹ️ RAG Explanation", "💬 App"])

# ----------- Tab 1: Explanation ----------- #
with tab1:
    st.subheader("If you haven't heard of RAG:")
    st.markdown("""
RAG (**Retrieval Augmented Generation**) is like **an AI with specific knowledge**.  
While ChatGPT, Gemini, and others use general information, with RAG you define the sources,  
and the AI resolves your prompts using *only those sources*.
""")
    image_path = os.path.join(os.path.dirname(__file__), "RAGvs.png")
    im = Image.open(image_path)
    st.image(im, caption="RAG vs General Purpose AI", use_container_width=True)

# ----------- Tab 2: App ----------- #
with tab2:
    st.subheader("Ask a Question")
    in_text = st.text_input("Insert your prompt")

    if st.button("Query"):
        if not in_text.strip():
            st.error("⚠️ You must provide a query.")
        else:
            if "llm" not in st.session_state or "vectordb" not in st.session_state:
                st.error("⚠️ You must process URLs first in the sidebar.")
            else:
                with st.spinner("Prompting the query... Please wait."):
                    try:
                        answer = rag.qa_prediction(
                            in_text, st.session_state.llm, st.session_state.vectordb
                        )
                        st.success("✅ Query successfully processed")
                        st.markdown(f"### 📝 Answer:\n{answer}")
                    except Exception as e:
                        st.error(f"❌ Error resolving the query: {str(e)}")
