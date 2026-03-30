from student_model.profile import StudentProfile
from utils.helpers import softmax

def update_profile(profile: StudentProfile, user_message: str) -> StudentProfile:
    DECAY = 0.8

    profile.asks_exercise *= DECAY
    profile.asks_detail *= DECAY
    profile.asks_objectivity *= DECAY
    
    text = user_message.lower()

    if "exercise" in text or "practice" in text:
    profile.asks_exercise += 1

    if "detail" in text or "example" in text:
        profile.asks_detail += 1

    if "summarize" in text or "direct" in text:
        profile.asks_objectivity += 1

    scores = {
        "analytical": profile.asks_detail,
        "explorer": profile.asks_exercise,
        "objective": profile.asks_objectivity,
    }

    probs = softmax(scores)

    profile.current_profile = max(probs, key=probs.get)
    profile.confidence = probs[profile.current_profile]

    return profile
