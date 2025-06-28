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

# --- 工具定义 ---

# 1. 知识库工具：基于文档的检索问答系统
def setup_knowledge_base_tool(pdf_path: str):
    """
    设置知识库工具，从 PDF 文档中检索信息。
    如果 PDF 文件不存在，则返回一个占位符工具。
    """
    if not os.path.exists(pdf_path):
        print(f"警告：未找到知识库文档 '{pdf_path}'。知识库工具将无法正常工作。")
        return Tool(
            name="KnowledgeBaseQueryTool",
            description="知识库工具 (未配置文档)。当你需要从项目文档或特定知识库中获取信息时使用此工具。输入是你想要查询的问题。",
            func=lambda x: f"错误：知识库文档 '{pdf_path}' 未找到，请联系管理员配置。"
        )
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        return Tool(
            name="KnowledgeBaseQueryTool",
            description="当你需要从项目文档或特定知识库中获取信息时使用此工具。输入是你想要查询的问题。",
            func=qa_chain.run
        )
    except Exception as e:
        print(f"知识库工具初始化失败: {e}")
        return Tool(
            name="KnowledgeBaseQueryTool",
            description="知识库工具 (初始化失败)。当你需要从项目文档或特定知识库中获取信息时使用此工具。输入是你想要查询的问题。",
            func=lambda x: f"错误：知识库工具初始化失败: {e}"
        )
