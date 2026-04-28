import agent.prompts as prompts
from agent.state import TutorState, TutorConfig
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from student_model.updater import update_student_profile, update_conversation_topic

def decide(state: TutorState, config: TutorConfig, model):
    """Decides whether to call the retrieval tool or generate direct answer
    based on the current conversation state."""

    system_prompt = prompts.DECIDE_PROMPT.format(domain=config.subject)

    response = (
        model.invoke(
            [SystemMessage(system_prompt)] + state["messages"]
        )
    )

    return {"messages": [response], "student_profile": state["student_profile"]}

def generate_answer(state: TutorState, config: TutorConfig, model):
    """Generates an answer based on the current conversation state and student profile."""
    
    # utlima mensagem do usuário
    question = next(
        msg.content for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    )
    # ultima resposta do retriever
    context = next(
        msg.content for msg in reversed(state["messages"])
        if isinstance(msg, ToolMessage)
    ) 
    
    # ------ geração --------
    system_prompt = SystemMessage(
        content=prompts.GENERATE_PROMPT.format(
            domain=config.subject,
            question=question,
            context=context
        )
    )

    response = model.invoke(
        [system_prompt] + state["messages"]
    )

    # # ------ self-check --------
    # check_prompt = SystemMessage(
    #     content=prompts.SELF_CHECK_PROMPT.format(
    #         answer=response.content,
    #         question=question,
    #         context=context
    #     )
    # )

    # check_response = model.invoke([check_prompt])

    # # ------ decisao --------
    # if check_response.content.strip() != "OK" or "OK" not in check_response.content.strip():
    #     final_response = AIMessage(content=check_response.content)
    # else:
    #     final_response = response

    final_response = response

    return {"messages": [final_response], "student_profile": state["student_profile"]}

def update_tracking(state: TutorState):
    """Updates the student profile and topic based on the conversation history."""
    
    last_user_msg = next(
        msg.content for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    )

    profile = update_student_profile(state.get("student_profile"), last_user_msg)
    #topic = update_conversation_topic(state.get("current_topic"), last_user_msg)

    return {"messages": state["messages"], "student_profile": profile, "current_topic": None}