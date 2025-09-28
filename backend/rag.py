#%% Library importation

#-------------- Langchain API
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader,WebBaseLoader  # ✅ removed SeleniumURLLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import JsonOutputParser
from fastapi import HTTPException
from backend.log_setup import logger_setup
import os
import pathlib

#-------------- Logger
logger = logger_setup("logger_setup", "server.log")

#-------------- Env libraries
from dotenv import load_dotenv
from pathlib import Path

#-------------- Other
from datetime import datetime
from uuid import uuid4  


#%% Global variables and env variables loading

#RAG parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 20
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "my_first_RAG"
PATH = "./vector_store"

#Loading environment variables
load_dotenv(override=False)


#%% VECTOR DATABASE Class DEFINITION
class VECTORDB_SYSTEM:
    def __init__(self, collection_name, path_dir, ef):
        self.chroma_db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=path_dir,
            embedding_function=ef
        )

    def clean_collection(self):
        self.chroma_db.reset_collection()

    def add_docs(self, list_urls):
        # Try Unstructured first (with UA + tolerant failures)
        ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"}
        loader = UnstructuredURLLoader(
            urls=list_urls,
            continue_on_failure=True,
            headers=ua,
        )
        docs = loader.load()
    
        # Filter out empty page_content
        docs = [d for d in docs if getattr(d, "page_content", "").strip()]
        if not docs:
            logger.warning("Unstructured returned no content. Falling back to WebBaseLoader.")
            # Minimal fallback using BeautifulSoup-based loader
            wb = WebBaseLoader(web_paths=list_urls, header_template=ua, verify_ssl=True)
            docs = wb.load()
            docs = [d for d in docs if getattr(d, "page_content", "").strip()]
    
        if not docs:
            logger.error("No content could be extracted from provided URLs.")
            raise HTTPException(status_code=400, detail="No content could be extracted from the provided URLs")
    
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
    
        if not chunks:
            logger.error("Document splitting returned 0 chunks.")
            raise HTTPException(status_code=400, detail="Unable to split documents into chunks")
    
        logger.info(f"Documents split into {len(chunks)} chunks")
        self.chroma_db.add_documents(
            chunks, ids=[str(uuid4()) for _ in range(len(chunks))]
        )
        logger.info("Documents added to Chroma database") 


#%% Function definition

def initializer():
    '''Initialize LLM + vector database'''
    groq_key_api = os.getenv("GROQ_API_KEY")
    if not groq_key_api:
        raise RuntimeError("GROQ Key is not set in the environment")

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.8,
        max_tokens=500,
        api_key=groq_key_api
    )

    emb_fun = HuggingFaceEmbeddings(
        model_name=EMBEDDING_NAME,
        model_kwargs={"trust_remote_code": True}
    )

    vectordb = VECTORDB_SYSTEM(COLLECTION_NAME, PATH, emb_fun)
    vectordb.clean_collection()
    return llm, vectordb


def populate_db(vectordb, list_urls):
    logger.info(f"{list_urls}")
    url_validation = [url for url in list_urls if url != ""]
    if len(url_validation) > 0:
        vectordb.add_docs(list_urls)
        return {"Status": f"{len(url_validation)} URLS provided"}
    else:
        raise HTTPException(status_code=400, detail="No valid URLs provided")


def qa_prediction(in_text: str, llm, vectordb):
    temp_folder= Path(__file__).resolve().parent / "template.txt"
    with open(temp_folder, "r", encoding="utf-8") as f:
        template = f.read()

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["input", "context"]
    )

    doc_template = PromptTemplate(
        template="Content:{page_content}\nSource:{source}",
        input_variables=["page_content", "source"]
    )

    retriever = vectordb.chroma_db.as_retriever(search_kwargs={"k": 5})

    qa_chain = create_stuff_documents_chain(llm, prompt_template, document_prompt=doc_template)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    result = rag_chain.invoke({"input": in_text}) 
    parser = JsonOutputParser()
    answer = parser.parse(result["answer"])

    return answer
