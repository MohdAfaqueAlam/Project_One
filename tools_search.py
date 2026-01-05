from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

tool_with_search = llm.bind_tools([search_tool])

result = tool_with_search.invoke("Who is Ada Lovelace and what is she known for?")
print(result.tool_calls)
