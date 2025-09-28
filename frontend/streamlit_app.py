import streamlit as st
from PIL import Image
import requests

API_URL = "http://localhost:8000"

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
            with st.spinner("Processing URLs... Please wait."):
                response = requests.post(f"{API_URL}/filereading", json={"list_urls": urls})
            if response.status_code == 200:
                raw = response.json()
                status_msg = raw.get("Status", "No status returned")
                st.success(f"✅ Successfully processed the URLs | {status_msg}")
            else:
                st.error("❌ Failed to process URLs")

            # Optional: Inspection call (not displayed, but ensures DB is ready)
            requests.get(f"{API_URL}/inspection")

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
    im = Image.open("RAGvs.png")
    st.image(im, caption="RAG vs General Purpose AI", use_container_width=True)

# ----------- Tab 2: App ----------- #
with tab2:
    st.subheader("Ask a Question")
    in_text = st.text_input("Insert your prompt")

    if st.button("Query"):
        if not in_text.strip():
            st.error("⚠️ You must provide a query.")
        else:
            with st.spinner("Prompting the query... Please wait."): 
                response = requests.post(f"{API_URL}/filereading/queryanswer", json={"in_text": in_text})
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "No answer provided")
                source = result.get("sources", "No sources provided")

                st.success("✅ Query successfully processed")
                st.markdown(f"### 📝 Answer:\n{answer}")
                st.markdown(f"### 📚 Source(s):\n{source}")
            else:
                st.error("❌ Error resolving the query")
