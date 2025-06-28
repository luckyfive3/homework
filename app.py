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

# 2. API 工具：天气查询
def get_current_weather(location: str) -> str:
    """
    获取指定地点的当前天气信息。
    """
    if not OPENWEATHERMAP_API_KEY:
        return "OpenWeatherMap API Key 未配置，无法查询天气。"

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric" # 获取摄氏度
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data.get("cod") == 200:
            main_info = data.get("main", {})
            weather_info = data.get("weather", [{}])[0]
            wind_info = data.get("wind", {})

            return (
                f"Location: {data.get('name')}, {data.get('sys', {}).get('country')}\n"
                f"Temperature: {main_info.get('temp')}°C\n"
                f"Feels Like: {main_info.get('feels_like')}°C\n"
                f"Humidity: {main_info.get('humidity')}%\n"
                f"Weather: {weather_info.get('description')}\n"
                f"Wind Speed: {wind_info.get('speed')} m/s"
            )
        else:
            return f"无法获取 {location} 的天气信息: {data.get('message', '未知错误')}"
    except requests.exceptions.RequestException as e:
        return f"查询天气时发生网络错误: {e}"
    except json.JSONDecodeError:
        return "API响应格式错误，无法解析天气信息。"
    except Exception as e:
        return f"查询天气时发生未知错误: {e}"

# 3. 本地函数工具：数据排序分析
def analyze_and_sort_data(data_str: str, sort_order: str = "ascending") -> str:
    """
    接收一个逗号分隔的数字字符串，将其转换为列表，进行排序，并提供基本统计分析。
    Args:
        data_str (str): 逗号分隔的数字字符串，例如 "10,5,20,15,8"。
        sort_order (str): 排序顺序，可选 "ascending" (升序) 或 "descending" (降序)。
    Returns:
        str: 排序后的数据和基本统计信息。
    """
    try:
        data_list = [float(x.strip()) for x in data_str.split(',')]
    except ValueError:
        return "输入数据格式不正确，请提供逗号分隔的数字。"

    if not data_list:
        return "输入数据为空。"

    if sort_order.lower() == "ascending":
        sorted_data = sorted(data_list)
    elif sort_order.lower() == "descending":
        sorted_data = sorted(data_list, reverse=True)
    else:
        return "排序顺序无效，请选择 'ascending' 或 'descending'。"

    df = pd.DataFrame(sorted_data, columns=["values"])
    description = df.describe().to_string() # 获取统计描述

    return (
        f"原始数据: {data_str}\n"
        f"排序方式: {sort_order}\n"
        f"排序后数据: {sorted_data}\n"
        f"统计分析:\n{description}"
    )
