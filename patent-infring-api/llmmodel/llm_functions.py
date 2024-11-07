from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from pymemcache.client import base
import openai
import numpy as np
import uuid,json
from transformers import pipeline


memcached_client = base.Client(('localhost', 11211))

# Configure OpenAI API key
openai.api_key = "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"
# Load Sentence-BERT model for similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define request and response schemas
class RequestBody(BaseModel):
    claims: List[str]
    feature: str
    similarity_threshold: float

class ResponseBody(BaseModel):
    feature_text: str
    combined_score: float
    abbreviated_summary: str
    explanation: str

# Initialize the summarization model
summarizer = pipeline("summarization", model="t5-small")

# Function to summarize a list of explanations into a paragraph
def summarize_explanations(explanations, max_length=150, min_length=50):
    # Concatenate explanations into a single text block
    combined_text = " ".join(explanations)

    # Use the summarizer to generate a paragraph
    summary = summarizer(combined_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Extract initial keywords dynamically using TF-IDF with bigrams
def extract_initial_keywords(claims, max_features=10):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(claims)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return list(feature_names)

# Calculate keyword presence score
def calculate_keyword_score(claim, keywords):
    matched_keywords = [keyword for keyword in keywords if keyword.lower() in claim.lower()]
    score = len(matched_keywords) / len(keywords)
    return score, matched_keywords

# Generate a claim summary using GPT-3.5-turbo
def generate_claim_summary(claim_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at summarizing patent claims."},
            {"role": "user", "content": f"Summarize the following patent claim in Concise and accurate expression rather than full sentence: {claim_text}"}
        ],
        max_tokens = 9,
        temperature = 0.3
    )
    summary = response['choices'][0]['message']['content'].strip()
    return summary

# Create a concise explanation using GPT-3.5-turbo with dynamic feature text
def create_readable_explanation(matched_keywords, claim_text, feature_description):
    # Construct the input prompt for GPT-3.5
    prompt = (
        f"The following is a patent claim:\n"
        f"'{claim_text}'\n"
        f"Explain in a concise manner how this claim relates to the product feature described as:\n"
        f"'{feature_description}'\n"
        f"Focus on keywords: {', '.join(matched_keywords)} and summarize in simple terms.",
        f"Additionally, provide a concise list of specific features or key components from the claim that align with the product description."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at explaining the relevance of patent claims concisely."},
            {"role": "user", "content": f"Briefly explain how this patent claim relates to the feature '{feature_description}', focusing on keywords: {', '.join(matched_keywords)}. The claim is: {claim_text}"}
        ],
        max_tokens=20,  # Reduce max tokens for shorter explanation
        temperature=0.3
    )
    output = response['choices'][0]['message']['content'].strip()
    # Split the output into explanation and specific features (assuming GPT-3.5 outputs them separately)
    if "Specific features:" in output:
        explanation, specific_features_section = output.split("Specific features:", 1)
        specific_features = [feature.strip() for feature in specific_features_section.split(",") if feature.strip()]
    else:
        explanation = output
        specific_features = []

    return explanation, specific_features

def analyze_claims_llm(body: RequestBody):
    claims = body.claims
    feature_description = body.feature
    similarity_threshold = body.similarity_threshold

    # Dynamically generate initial keywords from claims
    initial_keywords = extract_initial_keywords(claims)
    print(f"Initial Keywords Extracted: {initial_keywords}")
    
    keyword_weight = 0.4
    similarity_weight = 0.6

    # Encode the feature description
    feature_embedding = model.encode(feature_description, convert_to_tensor=True)

    # Process each claim and calculate relevance
    for claim in claims:
        # Encode claim
        claim_embedding = model.encode(claim, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity_score = util.pytorch_cos_sim(feature_embedding, claim_embedding).item()

        # Calculate keyword score
        keyword_score, matched_keywords = calculate_keyword_score(claim, initial_keywords)

        # Compute combined score
        combined_score = (keyword_weight * keyword_score) + (similarity_weight * similarity_score)

        if combined_score >= 0.5 and similarity_score >= similarity_threshold:
            # Generate summary and explanation for relevant claims using GPT-3.5-turbo
            claim_summary = generate_claim_summary(claim)
            explanation = create_readable_explanation(matched_keywords, claim, feature_description)

            # Prepare the response for the first relevant claim that meets threshold
            return {
                "feature_text": feature_description,
                "combined_score": combined_score,
                "abbreviated_summary": claim_summary,
                "explanation": explanation
            }

    # If no claims meet the threshold, return a default response
    return {
        "feature_text": feature_description,
        "combined_score": 0.0,
        "abbreviated_summary": "No claim met the similarity threshold.",
        "explanation": "No relevant claims were found based on the provided similarity threshold."
    }


# Helper function to perform infringement analysis
def perform_infringement_analysis_llm(patent, company):
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

        description_embedding = model.encode(description, convert_to_tensor=True)
        
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
            

            similarity_score = util.pytorch_cos_sim(description_embedding, claim_embedding).item()

            # Calculate keyword score
            keyword_score, matched_keywords = calculate_keyword_score(claim_text, initial_keywords)

            # Compute combined score
            combined_score = (keyword_weight * keyword_score) + (similarity_weight * similarity_score)

            if  combined_score >= 0.5 and similarity_score > 0.6:  # Threshold for similarity (can be adjusted)
                top_claims.append(claim_num)
                # Generate summary and explanation for relevant claims using GPT-3.5-turbo
                claim_summary = generate_claim_summary(claim_text)
                explanation, no_specific_features = create_readable_explanation(matched_keywords, claim, description)
                explanations.append(explanation) 
                specific_features.append(claim_summary)
                '''
                explanations.append(
                    f"Claim {claim_num} matches product feature with similarity score {similarity_score:.2f}."
                )
                '''

        if top_claims:
            infringement_likelihood = "High" if len(top_claims) > 5 else "Moderate"
            summary_paragraph = summarize_explanations(explanations)
            results.append({
                "product_name": product['name'],
                "infringement_likelihood": infringement_likelihood,
                "relevant_claims": top_claims,
                "explanation": summary_paragraph,
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