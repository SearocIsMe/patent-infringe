from pydantic import BaseModel

class AnlyzeRequest(BaseModel):
    patent_id: str
    company_name: str
    similarity_threshold: int


# Request body schema
class SimpleAnalysisRequest(BaseModel):
    patent_id: str
    company_name: str
    choose_gpt: bool
    fuzzy_logic_threshold: float
    dataset_type: bool
    similarity_threshold: float
