from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Hello World"}



class AnlyzeRequest(BaseModel):
    patent_id: str
    company_name: str
    similarity_threshold: int

@app.post("/analyze")
def analyze(request: AnlyzeRequest):
    print('Input request:' , request)
    return {
       "analysis_id": 123,
       "patent_id": "abc-2312312",
       "company_name": "this is fake company",
       "product_name": "infringed product",
       "infringement_likelihood": 0.96,
       "relevant_claims": ['2', '6', '12'],
       "explanation": "what we want to say, this is just a test",
       "specific_features":["item a", "item 2" ]
    }