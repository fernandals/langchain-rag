import agent.prompts as prompts
from agent.state import TutorState, TutorConfig
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

def decide(state: TutorState, config: TutorConfig, model):
    """Decides whether to call the retrieval tool or generate direct answer
    based on the current conversation state."""

    system_prompt = prompts.DECIDE_PROMPT.format(domain=config.subject)

    response = (
        model.invoke(
            [SystemMessage(system_prompt)] + state["messages"]
        )
    )

    return {"messages": [response], "profile": state["profile"]}

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

    return {"messages": [response], "profile": state["profile"]}
