from pydantic import BaseModel

class AnlyzeRequest(BaseModel):
    patent_id: str
    company_name: str
    similarity_threshold: int