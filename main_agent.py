import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.markdown import Markdown
from uuid import uuid4
from dotenv import load_dotenv
from langchain.mcp import MCPAdapter
import asyncio
import warnings
load_dotenv()
warnings.filterwarnings("ignore", message=".*Key '.*' is not supported in schema.*")

from row_reduction_agent import prompt_rr_agent

MEMORY_FILE = "ai_memory.md"
if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, 'w') as f:
        f.write("")
        print(f"Memory file created: {MEMORY_FILE}")

def main():
    asyncio.run(_main())

async def _main():
    config = {
        "mcpServers": {
            "docs-langchain": {
            "url": "https://docs.langchain.com/mcp"
            },
            "reference-langchain": {
            "url": "https://reference.langchain.com/mcp"
            },
        }   
    }

    async with MCPAdapter(config) as adapter:
        mcp_tools = await adapter.list_tools()

    console = Console()

    SYSTEM_PROMPT = f"""
    You are my personal ai assistant.

    The user can type 'exit' to finish their conversation.

    Use the update_memory tool to save important information, 
    however, before using the tool make sure to use the read_memory
    tool first so you can preserve any still relevent information.
    Make sure that the information in memory is useful and up-to-date.

    MEMORY: {open(MEMORY_FILE, 'r').read()}
    """

    model = init_chat_model(
        "google_genai:gemini-flash-lite-latest",
    )

    agent = create_agent(
        model=model,
        tools=(mcp_tools + [update_memory, read_memory, clear_memory, call_row_reduction_agent]),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )

    thread_config = {"configurable": {"thread_id": str(uuid4())}}
    while True:
        try:
            prompt = input("Input: ")
            if prompt == "exit": break
            response = await agent.ainvoke(
                {"messages": [HumanMessage(prompt)]},
                thread_config,
            )
        except EOFError:
            break
        console.print(Markdown(response["messages"][-1].text))

@tool
def update_memory(new_memory: str) -> str:
    """Replace all memory with a new full markdown state."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write(new_memory)
    return "Memory updated"

@tool
def read_memory() -> str:
    """Reads the entire memory file and returns its content."""
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Memory file not found."

@tool
def clear_memory() -> str:
    """Clears the entire memory file."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write("")
    return f"Successfully cleared memory file: {MEMORY_FILE}"

@tool
def call_row_reduction_agent(matrix: str) -> str:
    """Calls the row reduction agent.

    Give it the matrix that the user wants row reduced.
    The matrix should be formated as a list of lists.
    Make sure to show the user the matrix in a nicely formatted way.
    """
    result = prompt_rr_agent(matrix)

if __name__ == "__main__":
    main()