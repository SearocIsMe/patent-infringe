
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from datetime import datetime
from collections import Counter
import numpy as np
import uuid,json,re

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



# Helper function to extract specific features from the product description
def extract_specific_features(description, claims):
    # Tokenize and preprocess the description and claims
    words_in_description = re.findall(r'\b\w+\b', description.lower())
    feature_counter = Counter(words_in_description)

    # Extract keywords from the claims text to match against the description
    claim_keywords = set()
    for claim in claims:
        claim_text = claim['text']
        words_in_claim = re.findall(r'\b\w+\b', claim_text.lower())
        claim_keywords.update(words_in_claim)

    # Find specific features in the description that match claim keywords
    specific_features = [
        word for word in claim_keywords if word in feature_counter and feature_counter[word] > 1
    ]

    # Return the top specific features
    return specific_features[:10]  # Limit the number of features for readability


# Helper function to perform infringement analysis
def perform_infringement_analysis(patent, company, fuzzy_logic_threshold, similarity_threshold, memcached_client):
    results = []
    analysis_id = str(uuid.uuid4())
    date = datetime.now().strftime("%Y-%m-%d")

    # Parse the patent claims field correctly
    try:
        patent_claims = json.loads(patent['claims'])
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding patent claims")

    products = company['products']

    keyword_weight = fuzzy_logic_threshold
    similarity_weight = similarity_threshold

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

            if  combined_score >= 0.5 and similarity_score > similarity_threshold:  # Threshold for similarity (can be adjusted)
                top_claims.append(claim_num)
                # Generate explanation using NER and generative model
                explanation = create_readable_explanation(matched_keywords, claim_text)
                explanations.append(explanation)
                # Extract specific features from the description

                # this method is not accurate to give the features out as NER has such limitation
                # specific_features = extract_specific_features(description, patent_claims)
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