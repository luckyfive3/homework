import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain 相关的导入
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory # 弃用警告，但仍可使用
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# 其他工具所需的导入
import requests
import json
import pandas as pd

# --- 加载环境变量 ---
# 确保在 .env 文件中设置了 OPENAI_API_KEY 和 OPENWEATHERMAP_API_KEY
load_dotenv()

# --- 配置 API 密钥 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 环境变量未设置。请在 .env 文件中配置或通过 export 设置。")

# --- 初始化核心组件 ---
# 初始化 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# 初始化记忆模块
# LangChainDeprecationWarning: 请注意此处的弃用警告，未来可能需要根据官方指南迁移。
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
