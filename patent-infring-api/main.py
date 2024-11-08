from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pymemcache.client import base
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import json,uuid,re
import sys,os

PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(PROJECT_ROOT,'./basemodel'))
sys.path.insert(1, os.path.join(PROJECT_ROOT,'./nermodel'))
sys.path.insert(1, os.path.join(PROJECT_ROOT,'./llmmodel'))

from common import CacheSetUp
from request_body import AnlyzeRequest, SimpleAnalysisRequest
from ner_functions import (
    RequestBody, 
    ResponseBody,
    perform_infringement_analysis,
    generate_claim_summary,
    analyze_claims
    )
from llm_functions import (
    analyze_claims_llm, 
    perform_infringement_analysis_llm)

app = FastAPI()

# Load Sentence-BERT model for text similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

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


# Load data when the server starts
cacheMgr = CacheSetUp()

@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/query/{analysis_id}")
def queryAnalysis(analysis_id):
    print('Input analysisid:' , analysis_id)
    return {
       "analysis_id": "123",
       "patent_id": "abc-2312312",
       "company_name": "this is fake company",
       "analysis_date": "2024-10-31",
       "top_infringing_products": [ {
            "product_name": "Walmart Shopping App",
            "infringement_likelihood": 0.96,
            "relevant_claims": ['2', '6', '12'],
            "explanation": "what we want to say, this is just a test",
            "specific_features":["Direct advertisement-to-list functionality", "item 2" ]
        },
        {
            "product_name": "Walmart+",
            "infringement_likelihood": 0.96,
            "relevant_claims": ['2', '6', '12'],
            "explanation": "what we want to say, this is just a test",
            "specific_features":["Shopping list synchronization across devices", 
                                 "item 2" ]
        },
       ]
    }

@app.post("/analyze")
def analyze(request: AnlyzeRequest):
    print('Input request:' , request)
    return {
       "analysis_id": 123
       }

@app.post("/analyze_test_ner", response_model=ResponseBody)
async def analyze_claims_ner_api(body: RequestBody):
    return analyze_claims(body)

@app.post("/analyze_test_llm", response_model=ResponseBody)
async def analyze_claims_llm_api(body: RequestBody):
    return analyze_claims_llm(body)


# Endpoint for patent infringement analysis
@app.post("/analyze_infringement", response_model=Dict)
async def analyze_infringement(request: SimpleAnalysisRequest):

    # check if need to reload the source patent file.
    cacheMgr.reload(request.dataset_type)

    # Retrieve patent data from Memcached
    patent_data = cacheMgr.getPatent(f"patent:{request.patent_id}")

    if not patent_data:
        raise HTTPException(status_code=404, detail="Patent ID not found")

    patent = json.loads(patent_data)

    # Load company data directly from company.json
    company = cacheMgr.load_company_data(request.dataset_type, request.company_name)
    if not company:
        raise HTTPException(status_code=404, detail="Company name not found")

    # Perform infringement analysis
    if  request.choose_gpt:
        result = perform_infringement_analysis_llm(patent, company, request.fuzzy_logic_threshold, request.similarity_threshold, cacheMgr)
    else :
        result = perform_infringement_analysis(patent, company, request.fuzzy_logic_threshold, request.similarity_threshold, cacheMgr) 

    return result

@app.get("/get_analysis_result/{analysis_id}", response_model=Dict)
async def get_analysis_result(analysis_id: str):
    # Retrieve the analysis result from Memcached
    analysis_data = cacheMgr().getAnalysis(f"analysis:{analysis_id}")
    
    if not analysis_data:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    
    # Parse the retrieved data from JSON format
    analysis_result = json.loads(analysis_data)
    
    return analysis_result

