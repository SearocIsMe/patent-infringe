from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pymemcache.client import base
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sentence_transformers import SentenceTransformer, util

from queue import Queue

import json, uuid, re
import sys, os, time
import hashlib
import threading



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


# Initialize a queue with a maximum size of 500
QUEUE_MAX_SIZE = 500
analysis_queue = Queue(maxsize=QUEUE_MAX_SIZE)

# Function to generate a 12-character hash ID
def generate_hash_id(request_body: dict):
    # Combine all values into a single string
    hash_input = f"{request_body['patent_id']}-{request_body['company_name']}-" \
                 f"{request_body['choose_gpt']}-{request_body['fuzzy_logic_threshold']}-" \
                 f"{request_body['similarity_threshold']}-{request_body['dataset_type']}".encode('utf-8')
    
    # Generate SHA-256 hash and take the first 12 characters
    hash_output = hashlib.sha256(hash_input).hexdigest()
    return hash_output[:12]

# Worker function to process requests from the queue
def process_analysis_requests():
    while True:
        if not analysis_queue.empty():
            request = analysis_queue.get()
            try:

                # Perform infringement analysis
                if  request['choose_gpt']:
                    analysis_result = perform_infringement_analysis_llm(request['analysis_id'], request['patent'],
                        request['company'], request['fuzzy_logic_threshold'], request['similarity_threshold'], cacheMgr)
                else :
                    analysis_result = perform_infringement_analysis(request['analysis_id'], request['patent'],
                        request['company'], request['fuzzy_logic_threshold'], request['similarity_threshold'], cacheMgr)

                cacheMgr.set(f"analysis:{request['analysis_id']}", json.dumps(analysis_result))
            except Exception as e:
                print(f"Error processing request {request['analysis_id']}: {e}")
            analysis_queue.task_done()
        time.sleep(5)  # Prevent high CPU usage

# Start the worker thread
worker_thread = threading.Thread(target=process_analysis_requests, daemon=True)
worker_thread.start()


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

# for testing purpose
@app.post("/analyze_test_ner", response_model=ResponseBody)
async def analyze_claims_ner_api(body: RequestBody):
    return analyze_claims(body)

# for testing purpose
@app.post("/analyze_test_llm", response_model=ResponseBody)
async def analyze_claims_llm_api(body: RequestBody):
    return analyze_claims_llm(body)


# Endpoint for patent infringement analysis
@app.post("/analyze_infringement", response_model=Dict)
async def analyze_infringement(request: SimpleAnalysisRequest):
    # Generate a unique analysis ID based on hash
    analysis_id = generate_hash_id(request.__dict__)
    # check if need to reload the source patent file.
    cacheMgr.reload(request.dataset_type)

    # Retrieve patent data from Memcached
    patent_data = cacheMgr.getPatent(f"patent:{request.patent_id}")

    if not patent_data:
        return {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": "Patent ID not found"
        }

    patent = json.loads(patent_data)

    # Load company data directly from company.json
    company = cacheMgr.load_company_data(request.dataset_type, request.company_name)
    if not company:
        # Return the analysis ID and timestamp when the request is received
        return {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": "Company name not found"
        }
    
    # Check if the queue has reached its maximum size
    if analysis_queue.full():
        return {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": "The queue is full. Please try again later."
        
        }
    # Enqueue the analysis requesti
    try: 
        analysis_request = {
            "analysis_id": analysis_id,
            "patent": patent,
            "company": company,
            "fuzzy_logic_threshold": request.fuzzy_logic_threshold,
            "similarity_threshold": request.similarity_threshold,
            "choose_gpt": request.choose_gpt
        }
        analysis_queue.put_nowait(analysis_request)
    except Full:
        return {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": "The queue is full. Please try again later."
        }

    # Return the analysis ID and timestamp when the request is received
    return {
        "analysis_id": analysis_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "Request received and queued for processing"
    }


@app.get("/get_analysis_result/{analysis_id}", response_model=Dict)
async def get_analysis_result(analysis_id: str):
    # Retrieve the analysis result from Memcached
    analysis_data = cacheMgr.getAnalysis(f"analysis:{analysis_id}")
    
    if not analysis_data:
        return {
            "analysis_id": analysis_id,
            "error": "The analysis ID you provided is not found. It might still be processing or may not exist."
        }

    # Parse the retrieved data from JSON format
    analysis_result = json.loads(analysis_data)
    
    return analysis_result

