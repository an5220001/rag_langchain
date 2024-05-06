import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from src.base.llm_model import llm_model
from src.rag.main import build_rag_chain, InputQA, OutputQA
# from src.rag.main import get_answer, build_rag_chain, InputQA, OutputQA


llm = llm_model()
genai_docs = "./data_source/generative_ai"

# --------- Chains - - - - - - - - -- - - - - - -
genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

# --------- App - FastAPI ----------------
app = FastAPI(  
    title="Langchain Server",
    version="1.0",
    description="A simple api server using Langchian's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware ,
    allow_origins =["*"] ,
    allow_credentials =True ,
    allow_methods =["*"] ,
    allow_headers =["*"] ,
    expose_headers =["*"] ,
)

# --------- Routes - FastAPI ----------------
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(input: InputQA):
    answer = genai_chain.invoke(input.question)
    return {"answer": answer}

# --------- Langserve Routes - Playground ----------------
add_routes(app,
           genai_chain,
           playground_type="default",
           path="/generative_ai")