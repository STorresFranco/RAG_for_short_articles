#%% Library importation

#-------------- Langchain API
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import JsonOutputParser
from backend.log_setup import logger_setup

#-------------- Other
from datetime import datetime
from uuid import uuid4  

#-------------- Logger
logger=logger_setup("logger_setup","server.log")

#-------------- Path libraries
from pathlib import Path



#%% Global variables and env variables loading

#RAG parameters
CHUNK_SIZE=500
CHUNK_OVERLAP=20
LLM_MODEL="llama-3.3-70b-versatile"
EMBEDDING_NAME="Qwen/Qwen3-Embedding-0.6B"
COLLECTION_NAME="my_first_RAG"
PATH="backend/vector_store"

#Loading environment variables
load_dotenv()


#%% VECTOR DATABASE Class DEFINITION: Works as an instance containing a chroma object from langchain
class VECTORDB_SYSTEM:
    def __init__(self,collection_name,path_dir,ef):
        self.chroma_db=Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=path_dir,
            embedding_function=ef
        )

    def clean_collection(self):
        self.chroma_db.reset_collection()

    def add_docs(self,list_urls):
        wb = WebBaseLoader(
            list_urls,
            header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"})
        docs = wb.load()
        splitter=RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", "<p>", "<div>", "</p>", "</div>"],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks=splitter.split_documents(docs)
        logger.info(f"Documents splited into {len(chunks)} chunks")
        logger.info(f"First Chunk preview {chunks[0].page_content[:300]}")
        self.chroma_db.add_documents(chunks,ids=[str(uuid4()) for _ in range(len(chunks))]) 
        logger.info("Documents added to Chroma database")   

#%% Function definition

def initializer():
    ''' 
    Description: 
        Function used to initialize and llm and a vector database with a collection based on user interaction with frontend
        The function will be called based on frontend "Begin" button
    '''
    #Creating the llm model from Groq
    llm=ChatGroq(model=LLM_MODEL,
                 temperature=0.8,
                 max_tokens=500,
                )

    #Calling the embedding function from huggingface
    emb_fun=HuggingFaceEmbeddings(
        model_name=EMBEDDING_NAME,
        model_kwargs={"trust_remote_code":True}
    )

    #Initializing the vector database instance
    vectordb=VECTORDB_SYSTEM(COLLECTION_NAME,PATH,emb_fun) #Initializing the instance
    vectordb.clean_collection()                            #Cleaning the collection
    return llm,vectordb

def populate_db(vectordb,list_urls):
    '''
    Description
        Function to populate the vector database with chunks based on the URLs provided by the user in the frontend
    Inputs
        vectordb (RAG_SYSTEM instance): instance of RAG_SYSTEM class
        list_urls (list): list of urls taken from frontend
    '''
    #adding the documents from the url list to the database
    logger.info(f"{list_urls}")
    url_validation=[url for url in list_urls if url != ""]
    if len(url_validation)>0:
        vectordb.add_docs(list_urls)
        return {"Status":f"{len(url_validation)} URLS provided"}
    else:
        raise HTTPException(status_code=400, detail="No valid URLs provided")

def qa_prediction(in_text:str,llm,vectordb):
    ''' 
    Description
        Function to predict an answer based on a user query using stored documents in a Chroma Database using an llm model.
    Inputs
        in_text (str): Query provided by the user
        llm (ChatGroq instance): llm model from GROQ
        vectordb (VECTORDB_SYSTEM instance): Instance of VECTORDB_SYSTEM class
    '''
    #Create prompt template
    with open("backend/template.txt","r", encoding="utf-8") as f:
        template=f.read() #Loading the template from txt file

    prompt_template=PromptTemplate(
        template=template,
        input_variables=["input","context"]
    )

    #Create the document template as the structure passed to the llm
    doc_template=PromptTemplate(
        template="Content:{page_content}\nSource:{source}",
        input_variables=["page_content","source"]
    )

    #Creating the retriever
    retriever=vectordb.chroma_db.as_retriever(search_kwargs={"k":5})

    #Creating the prompt chain and retrieval chain
    qa_chain=create_stuff_documents_chain(llm,prompt_template,document_prompt=doc_template)
    rag_chain=create_retrieval_chain(retriever,qa_chain)

    #Taking the results and parsing the json format
    result=rag_chain.invoke({"input":in_text}) 
    parser=JsonOutputParser()
    answer=parser.parse(result["answer"])

    return answer





