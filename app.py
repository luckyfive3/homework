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

# --- 实例化工具 ---
# 知识库文档路径，请确保 'docs/your_project_document.pdf' 存在
KNOWLEDGE_BASE_PDF_PATH = "./docs/your_project_document.pdf"
knowledge_tool = setup_knowledge_base_tool(KNOWLEDGE_BASE_PDF_PATH)

weather_tool = Tool(
    name="WeatherQueryTool",
    description="当你需要查询某个城市的当前天气情况时使用此工具。输入是城市的名称，例如 'Beijing'。",
    func=get_current_weather
)

data_analysis_tool = Tool(
    name="DataAnalysisAndSortTool",
    description="当你需要对一系列逗号分隔的数字进行排序和基本统计分析时使用此工具。输入是数字字符串，可选参数 'sort_order' 可以是 'ascending' 或 'descending'。",
    func=analyze_and_sort_data
)

# 将所有工具汇集
tools = [knowledge_tool, weather_tool, data_analysis_tool]

# --- Agent 定义 ---
# *** 重点修改此处：添加 {tool_names} 占位符 ***
main_agent_prompt_template = PromptTemplate.from_template("""
你是一个智能助理 Agent，能够理解用户的请求，并利用我提供的工具来完成任务。
你的目标是尽可能准确和全面地回答用户的问题，并协助他们完成各项任务。

可用的工具包括：{tool_names}。
你有权访问以下工具: {tools}
---
对话历史:
{chat_history}
---
用户请求: {input}
---
{agent_scratchpad}
""")

# 创建 AgentExecutor
# create_react_agent 会根据 tools 列表自动填充 {tool_names}
agent_runnable = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=main_agent_prompt_template.partial(tools=tools) # 这里依然传入 tools
)

main_agent = AgentExecutor(
    agent=agent_runnable,
    tools=tools,
    verbose=True, # 开启 verbose 可以看到 Agent 的思考过程
    memory=memory, # 传入记忆模块
    handle_parsing_errors=True # 允许 Agent 尝试处理工具调用时的解析错误
)

# --- FastAPI 应用 ---
app = FastAPI(
    title="协作型智能 Agent 系统",
    description="一个基于 LangChain 的智能 Agent，能够通过对话形式协助用户完成任务。",
    version="1.0.0"
)

# 请求体模型
class UserQuery(BaseModel):
    query: str

@app.post("/chat", summary="与智能 Agent 进行对话", response_description="Agent 的回复")
async def chat_with_agent(user_query: UserQuery):
    """
    通过 POST 请求与智能 Agent 进行对话。
    Agent 会利用其工具和记忆来理解并响应用户查询。
    """
    try:
        response = await main_agent.ainvoke({"input": user_query.query})
        return {"response": response["output"]}
    except Exception as e:
        print(f"Agent 处理请求时发生错误: {e}")
        return {"error": f"Agent 处理请求时发生错误: {e}. 请稍后再试或检查日志。"}

# --- 运行服务器 ---
if __name__ == "__main__":
    # 使用 uvicorn 运行 FastAPI 应用
    # host="0.0.0.0" 表示监听所有网络接口，可以在局域网内访问
    # port=8000 是默认端口，可以修改
    uvicorn.run(app, host="0.0.0.0", port=8000)
