from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json,uuid
import sys,os
from datetime import datetime
from pymemcache.client import base
from typing import List, Dict

PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(PROJECT_ROOT,'./basemodel'))
sys.path.insert(1, os.path.join(PROJECT_ROOT,'./nermodel'))
sys.path.insert(1, os.path.join(PROJECT_ROOT,'./llmmodel'))

from request_body import AnlyzeRequest
from ner_functions import RequestBody, ResponseBody, extract_initial_keywords, calculate_keyword_score, create_readable_explanation, analyze_claims,generate_claim_summary
from llm_functions import analyze_claims_llm

from sentence_transformers import SentenceTransformer, util


app = FastAPI()
memcached_client = base.Client(('localhost', 11211))

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

'''
class AnlyzeRequest(BaseModel):
    patent_id: str
    company_name: str
    similarity_threshold: int
'''

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

@app.post("/analyze_claims_ner", response_model=ResponseBody)
async def analyze_claims_ner_api(body: RequestBody):
    return analyze_claims(body)

@app.post("/analyze_claims_llm", response_model=ResponseBody)
async def analyze_claims_llm_api(body: RequestBody):
    return analyze_claims_llm(body)


# Load and cache data into Memcached
def load_data_into_memcached():
    # Load patent source data
    with open('./data/patent_source.json', 'r') as file:
        patent_data = json.load(file)
        for patent in patent_data:  # Iterate directly over the array of sub-JSONs
            memcached_client.set(f"patent:{patent['publication_number']}", json.dumps(patent))
    

# Helper function to load company data directly from company.json
def load_company_data(company_name):
    with open('./data/company.json', 'r') as file:
        company_data = json.load(file)
        for company in company_data['companies']:
            if company['name'].lower() == company_name.lower():
                return company
    return None

# Load data when the server starts
load_data_into_memcached()

# Request body schema
class AnalysisRequest(BaseModel):
    patent_id: str
    company_name: str
# Helper function to perform infringement analysis
def perform_infringement_analysis(patent, company):
    results = []
    analysis_id = str(uuid.uuid4())
    date = datetime.now().strftime("%Y-%m-%d")

    # Parse the patent claims field correctly
    try:
        patent_claims = json.loads(patent['claims'])
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding patent claims")

    products = company['products']

    keyword_weight = 0.4
    similarity_weight = 0.6

    for product in products:
        description = product['description']
        top_claims = []
        explanations = []
        specific_features = []


        for claim in patent_claims:
            claim_num = claim['num']
            claim_num.lstrip("0")
            claim_text = claim['text']

            claim_array = []
            claim_array.append(claim_text);
            # Dynamically generate initial keywords from claims
            initial_keywords = extract_initial_keywords(claim_array)
            print(f"Initial Keywords Extracted: {initial_keywords}")

            claim_embedding = model.encode(claim_text, convert_to_tensor=True)
            description_embedding = model.encode(description, convert_to_tensor=True)

            similarity_score = util.pytorch_cos_sim(description_embedding, claim_embedding).item()

            # Calculate keyword score
            keyword_score, matched_keywords = calculate_keyword_score(claim_text, initial_keywords)

            # Compute combined score
            combined_score = (keyword_weight * keyword_score) + (similarity_weight * similarity_score)

            if  combined_score >= 0.5 and similarity_score > 0.6:  # Threshold for similarity (can be adjusted)
                top_claims.append(claim_num)
                # Generate explanation using NER and generative model
                explanation = create_readable_explanation(matched_keywords, claim_text)
                explanations.append(explanation) 
                specific_features.append(generate_claim_summary(claim_text))
                '''
                explanations.append(
                    f"Claim {claim_num} matches product feature with similarity score {similarity_score:.2f}."
                )
                '''

        if top_claims:
            infringement_likelihood = "High" if len(top_claims) > 5 else "Moderate"
            results.append({
                "product_name": product['name'],
                "infringement_likelihood": infringement_likelihood,
                "relevant_claims": top_claims,
                "explanation": " ".join(explanations),
                "specific_features": specific_features  # Placeholder: replace with actual feature extraction if needed
            })

    # Sort and return top two products
    results = sorted(results, key=lambda x: len(x['relevant_claims']), reverse=True)[:2]

    overall_risk_assessment = (
        "High risk of infringement due to implementation of core patent claims in multiple products."
        if any(res['infringement_likelihood'] == "High" for res in results)
        else "Moderate risk due to partial implementation of patented technology."
    )

    # Create result structure
    analysis_result = {
        "analysis_id": analysis_id,
        "patent_id": patent['publication_number'],
        "company_name": company['name'],
        "analysis_date": date,
        "top_infringing_products": results,
        "overall_risk_assessment": overall_risk_assessment
    }

    # Store the result in Memcached
    memcached_client.set(f"analysis:{analysis_id}", json.dumps(analysis_result))

    return analysis_result

# Endpoint for patent infringement analysis
@app.post("/analyze_infringement", response_model=Dict)
async def analyze_infringement(request: AnalysisRequest):
    # Retrieve patent data from Memcached
    patent_data = memcached_client.get(f"patent:{request.patent_id}")
    if not patent_data:
        raise HTTPException(status_code=404, detail="Patent ID not found")

    patent = json.loads(patent_data)

    # Load company data directly from company.json
    company = load_company_data(request.company_name)
    if not company:
        raise HTTPException(status_code=404, detail="Company name not found")

    # Perform infringement analysis
    result = perform_infringement_analysis(patent, company)

    return result

@app.get("/get_analysis_result/{analysis_id}", response_model=Dict)
async def get_analysis_result(analysis_id: str):
    # Retrieve the analysis result from Memcached
    analysis_data = memcached_client.get(f"analysis:{analysis_id}")
    
    if not analysis_data:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    
    # Parse the retrieved data from JSON format
    analysis_result = json.loads(analysis_data)
    
    return analysis_result