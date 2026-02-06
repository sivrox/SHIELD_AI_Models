from pydantic import BaseModel

class ChatRequest(BaseModel):
    patient_id: int
    question: str


class ChatResponse(BaseModel):
    answer: str


class ReportResponse(BaseModel):
    message: str
    analysis: str
