from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

REACT_PROMPT = PromptTemplate.from_template("""
You are MedAssist AI, an intelligent medical research assistant.
You have access to the following tools:

{tools}

Use this EXACT format:

Question: the input question
Thought: what should I do?
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information
Final Answer: a clear, grounded, cited answer

Question: {input}
Thought: {agent_scratchpad}
""")


def build_agent(llm, tools: list[Tool]) -> AgentExecutor:
    """
    Build the ReAct agent executor.

    Args:
        llm   : LangChain LLM wrapper
        tools : List of Tool objects from build_tools()

    Returns:
        AgentExecutor ready to call with .invoke({"input": question})
    """
    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=4,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def query_agent(executor: AgentExecutor, question: str) -> dict:
    """
    Run a question through the agent and return a structured result.

    Returns:
        dict with keys: question, answer, steps, tools_used
    """
    result = executor.invoke({"input": question})
    steps = result.get("intermediate_steps", [])
    return {
        "question"  : question,
        "answer"    : result.get("output", ""),
        "steps"     : len(steps),
        "tools_used": [action.tool for action, _ in steps],
    }