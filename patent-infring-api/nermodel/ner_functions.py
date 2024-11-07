
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import numpy as np

# Load models
model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="t5-small", max_length=30, min_length=5, do_sample=False)
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
generator = pipeline("text2text-generation", model="t5-small")


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

# Extract initial keywords dynamically
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

# Generate automatic summary for a claim
def generate_claim_summary(claim_text):
    summary = summarizer(claim_text, max_length=20, min_length=5, do_sample=False)
    return summary[0]['summary_text']

# Generate a dynamic explanation using NER and a generative model
def create_readable_explanation(matched_keywords, claim_text):
    # Preprocess the claim text: simplify structure and remove special characters
    claim_text_processed = claim_text.replace(";", ".").replace(":", ".").strip()

    # Extract named entities using the NER model
    entities = ner(claim_text_processed)
    entity_descriptions = []

    for entity in entities:
        # Extract entity and its label
        entity_text = entity['word']
        entity_type = entity['entity']

        # Only consider entities related to functionalities or components (e.g., labeled as ORG, MISC)
        if entity_type in ["ORG", "MISC", "PRODUCT", "TECH", "PROCESS"]:
            # Use the generative model to describe the entity in context
            description = generator(f"Describe the functionality of {entity_text} in the context of {claim_text}", max_length=40)[0]['generated_text']
            entity_descriptions.append(description)

    # Concatenate the descriptions to form a coherent explanation
    core_explanation = "; ".join(entity_descriptions)

    # Final explanation based on matched keywords and generative model output
    explanation = (
        f"{core_explanation}. "
        f"'{generate_claim_summary(claim_text)}'."
    )
    return explanation

def analyze_claims(body: RequestBody):
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
            # Generate explanation using NER and generative model
            explanation = create_readable_explanation(matched_keywords, claim)

            # Prepare the response for the first relevant claim that meets threshold
            return {
                "feature_text": feature_description,
                "combined_score": combined_score,
                "abbreviated_summary": generate_claim_summary(claim),
                "explanation": explanation
            }

    # If no claims meet the threshold, return a default response
    return {
        "feature_text": feature_description,
        "combined_score": 0.0,
        "abbreviated_summary": "No claim met the similarity threshold.",
        "explanation": "No relevant claims were found based on the provided similarity threshold."
    }