from langgraph.graph import MessagesState
from student_model.profile import StudentProfile
from pydantic import BaseModel

class TutorConfig(BaseModel):
    subject: str
    allow_direct_answers: bool = False
    course_level: str = "beginner" # beginner, intermediate, advanced
    answer_language: str = "English"
    max_sentences: int = 6

class TutorState(MessagesState):
    profile: StudentProfile
    current_topic: str | None = None