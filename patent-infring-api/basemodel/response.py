from pydantic import BaseModel

from datetime import datetime, date, time


class TopInfringingProducts(BaseModel):
    product_name: str
    infringement_likelihood: float
    relevant_claims: set[str] = set():
    explanation: str 
    specific_features: set[str] = set()


class AnalysisResult(BaseModel):
    analysis_id: str
    patent_id: str 
    company_name: str
    analysis_date: date 
    top_infringing_products: set[TopInfringingProducts] = set()
