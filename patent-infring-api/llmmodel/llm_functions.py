from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import openai
import numpy as np

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
            {"role": "user", "content": f"Summarize the following patent claim in one sentence: {claim_text}"}
        ],
        max_tokens=15,
        temperature=0.3
    )
    summary = response['choices'][0]['message']['content'].strip()
    return summary

# Create a concise explanation using GPT-3.5-turbo with dynamic feature text
def create_readable_explanation(matched_keywords, claim_text, feature_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at explaining the relevance of patent claims concisely."},
            {"role": "user", "content": f"Briefly explain how this patent claim relates to the feature '{feature_text}', focusing on keywords: {', '.join(matched_keywords)}. The claim is: {claim_text}"}
        ],
        max_tokens=20,  # Reduce max tokens for shorter explanation
        temperature=0.3
    )
    explanation = response['choices'][0]['message']['content'].strip()
    return explanation

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