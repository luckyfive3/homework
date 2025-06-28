import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

import requests
import json
import pandas as pd

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 环境变量未设置。请在 .env 文件中配置或通过 export 设置。")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
def setup_knowledge_base_tool(pdf_path: str):
    # ... (与之前提供的代码相同，包含错误处理和FAISS/OpenAI Embeddings)
    # 确保PDF文件存在，否则返回错误提示工具
def analyze_and_sort_data(data_str: str, sort_order: str = "ascending") -> str:
    # ... (与之前提供的代码相同，包含数据处理和错误处理)
    # 使用 Python 内置函数和 pandas 进行数据排序和统计
KNOWLEDGE_BASE_PDF_PATH = "./docs/your_project_document.pdf"
knowledge_tool = setup_knowledge_base_tool(KNOWLEDGE_BASE_PDF_PATH)
weather_tool = Tool(...) # 实例化天气工具
data_analysis_tool = Tool(...) # 实例化数据分析工具

tools = [knowledge_tool, weather_tool, data_analysis_tool]
main_agent_prompt_template = PromptTemplate.from_template("""
你是一个智能助理 Agent，能够理解用户的请求，并利用我提供的工具来完成任务。
你的目标是尽可能准确和全面地回答用户的问题，并协助他们完成各项任务。
你有权访问以下工具: {tools}
---
对话历史:
{chat_history}
---
用户请求: {input}
---
{agent_scratchpad}
""")
agent_runnable = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=main_agent_prompt_template.partial(tools=tools)
)
main_agent = AgentExecutor(
    agent=agent_runnable,
    tools=tools,
    verbose=True, # 开启 verbose 可以看到 Agent 的思考过程
    memory=memory, # 传入记忆模块
    handle_parsing_errors=True
)
app = FastAPI(
    title="协作型智能 Agent 系统",
    description="一个基于 LangChain 的智能 Agent，能够通过对话形式协助用户完成任务。",
    version="1.0.0"
)
class UserQuery(BaseModel):
    query: str
  @app.post("/chat", summary="与智能 Agent 进行对话", response_description="Agent 的回复")
async def chat_with_agent(user_query: UserQuery):
    try:
        response = await main_agent.ainvoke({"input": user_query.query})
        return {"response": response["output"]}
    except Exception as e:
        print(f"Agent 处理请求时发生错误: {e}")
        return {"error": f"Agent 处理请求时发生错误: {e}. 请稍后再试或检查日志。"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
