#%% Library importation
from fastapi import FastAPI
import rag
from pydantic import BaseModel
from typing import List,Optional
from contextlib import asynccontextmanager
from log_setup import logger_setup

logger2=logger_setup("logger_setup","server.log")
#%%Creating the server

#Initialization 
llm,vectordb=None,None
@asynccontextmanager
async def lifespan(server:FastAPI):
    global llm,vectordb
    llm,vectordb=rag.initializer()
    tp=type(vectordb)
    logger2.info(tp)
    print("Vector database and LLM initialization at startup")
    yield


server=FastAPI(lifespan=lifespan)

#%% Pydantic Basemodels
#Class to validate the url list
class UrlPayload(BaseModel): 
    list_urls:List[str]

#Class to validate the query as text
class QueryPayload(BaseModel):
    in_text:str

#%%REST points
#Doc adding
@server.post("/filereading")
def doc_population(payload:UrlPayload):
    print(payload.model_dump_json())
    status=rag.populate_db(vectordb,payload.list_urls)
    logger2.info(vectordb)
    return status

#Inspection for debug
@server.get("/inspection")
def chroma_inspection():
    if not vectordb:
        logger2.error("vectordb is not initialized")
        return {"Status":"vectordb is not initialized"}
    else:
        docs_info=vectordb.chroma_db.get()
        vectordb_inspection=  {
        "Status":"ok",
        "num_docs":len(docs_info.get("documents",[])),
        "ids":docs_info.get("ids",[])[:5]
        }
        logger2.info(vectordb_inspection)


#Query response
@server.post("/filereading/queryanswer")
def query_response(payload:QueryPayload):
    answer=rag.qa_prediction(payload.in_text,llm,vectordb)
    return answer
