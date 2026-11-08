# Retrieval-Augmented Generation (RAG) for Short Articles

A lightweight **Retrieval-Augmented Generation (RAG)** application built with Python, LangChain, ChromaDB, Groq, and Streamlit.

The application allows users to provide up to three article URLs and ask questions about their content. Instead of relying exclusively on the LLM's general knowledge, the system retrieves relevant information from the supplied documents to generate concise, context-based answers with source references.

The project was developed as a practical implementation of the RAG architecture, covering document ingestion, text embeddings, vector similarity search, and LLM-based answer generation.

## How It Works

The application follows a standard RAG workflow:

1. **Document ingestion:** WebBaseLoader extracts content from the URLs provided by the user.
2. **Text processing:** Documents are divided into smaller, overlapping chunks using RecursiveCharacterTextSplitter.
3. **Embedding generation:** Each chunk is transformed into a vector representation using the Qwen3-Embedding-0.6B model from Hugging Face.
4. **Vector storage:** Embeddings and document metadata are stored in a persistent ChromaDB collection.
5. **Similarity retrieval:** For each question, the system retrieves the five most relevant chunks from the vector database.
6. **Answer generation:** The retrieved context and the user's question are passed to a Groq-hosted LLM through LangChain. The response is returned in a structured JSON format containing an answer and source references.

## Tech Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| RAG orchestration | LangChain |
| Embedding model | Qwen3-Embedding-0.6B |
| Vector database | ChromaDB |
| LLM inference | Groq API |
| Document loading | WebBaseLoader |
| Text processing | RecursiveCharacterTextSplitter |

## Application Features

- Process content from up to three user-provided URLs.
- Ask natural-language questions about the processed articles.
- Retrieve relevant context through vector similarity search.
- Generate concise answers grounded in retrieved content.
- Display source references alongside generated answers.
- Explore a basic explanation of RAG through the Streamlit interface.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/STorresFranco/RAG_for_short_articles.git
cd RAG_for_short_articles
```

### 2. Install dependencies

A Python virtual environment is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Configure the Groq API

The application requires a Groq API key for LLM inference.

Set the `GROQ_API_KEY` environment variable before launching the application. For example, in PowerShell:

```powershell
$env:GROQ_API_KEY="your_api_key"
```

You can obtain an API key from [Groq Console](https://console.groq.com/).

The LLM and embedding model configurations are defined in `backend/rag.py`.

### 4. Run the application

From the repository's root directory:

```bash
streamlit run frontend/streamlit_app.py
```

Once the application starts:

1. Enter one to three article URLs in the sidebar.
2. Click **Process URLs** to extract and index their content.
3. Open the **App** tab and ask a question about the articles.
4. Review the generated answer and its source references.

## Project Structure

```text
RAG_for_short_articles/
├── backend/
│   ├── rag.py                # RAG pipeline and vector database
│   ├── template.txt          # LLM prompt template
│   ├── log_setup.py          # Logging configuration
│   └── vector_store/         # Persistent ChromaDB storage
├── frontend/
│   ├── streamlit_app.py      # Streamlit interface
│   └── Ragvs.png             # RAG explanation graphic
├── requirements.txt
└── README.md
```

## Implementation Details

The current configuration uses:

| Parameter | Value |
|---|---|
| Chunk size | 500 characters |
| Chunk overlap | 20 characters |
| Retrieved chunks | Top 5 |
| Embedding model | Qwen/Qwen3-Embedding-0.6B |
| LLM temperature | 0.8 |
| Maximum output tokens | 500 |

The vector collection is reset whenever new URLs are processed, allowing each indexing operation to focus on the latest documents.

## Limitations and Considerations

This project is intended as a practical RAG demonstration rather than a production-ready document intelligence system.

- **Document scope:** The application is optimized for relatively short articles. Processing large web pages may require more memory and CPU resources.
- **Hosting resources:** The embedding model runs locally within the application's environment, so initial model loading and document processing can be slow on limited cloud resources.
- **Web extraction:** Content retrieval depends on website accessibility and HTML structure.
- **Response reliability:** Generated answers depend on the quality of the retrieved context, and source references are generated by the LLM rather than independently verified.
- **Structured output:** The current implementation expects JSON-formatted responses from the model.

## Purpose

The goal of this project was to implement and understand the main components of a RAG system, from unstructured document ingestion to retrieval-based answer generation.

It serves as a foundation for exploring more advanced retrieval techniques, document processing strategies, and LLM-based applications.
