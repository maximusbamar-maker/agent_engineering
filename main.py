import os
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY is not set")

from langchain.messages import HumanMessage

# def get_response(prompt: str) -> str:
#     return prompt

# def main():
#     while True:
#         try:
#             prompt = input("Input: ")
#             if prompt == "exit": break
#             response = get_response(prompt)

#         except EOFError:
#             break
        
#         print(response)

# if __name__ == "__main__":
#     main()

from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

prompt = HumanMessage(input("Input: "))

result = agent.invoke(
    {"messages": [prompt]}
)
print(result["messages"][-1].text)