from pydantic import BaseModel

class StudentProfile(BaseModel):
    asks_exercise: int = 0
    asks_detail: int = 0
    asks_objectivity: int = 0

    current_profile: str = "neutral"
    confidence: float = 0.0