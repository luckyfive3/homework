# 协作型智能 Agent 系统

## 1. Agent 功能说明

### 1.1 项目背景与目标

本项目旨在于 LangChain 框架构建一个协作型智能 Agent 系统，并将其部署在服务器上。我们的核心目标是开发一个能够通过对话形式，智能地协助用户完成特定任务的 Agent。

该 Agent 系统能够：
- 准确理解用户的意图。
- 引用内部知识库内容以提供信息。
- 调用外部 API 获取实时数据（如天气）。
- 执行本地自定义函数进行数据处理（如排序分析）。
- 通过记忆模块保持对话上下文的连续性，实现多轮交互。

### 1.2 Agent 角色与协作

目前系统主要由一个核心的智能 Agent 构成。它充当了任务规划者和执行者的角色，能够根据用户请求，自主判断并选择合适的工具来完成任务。我们选择了 LangChain 的 ReAct Agent 模式，它通过“思考（Thought）-行动（Action）-观察（Observation）”的循环来驱动Agent的决策和执行。未来的优化方向将考虑引入更明确的协作 Agent 角色，例如一个“数据分析师 Agent”专注于数据处理，而“信息检索 Agent”专注于知识库查询，从而实现更专业的任务分派和执行。

## 2. 系统实现细节

### 2.1 使用的技术栈简介

* **核心框架：** LangChain (用于 Agent、工具、记忆模块的构建和编排)
* **语言模型 (LLM)：** OpenAI 的 `gpt-4o-mini` (通过 `langchain-openai` 集成)
* **Web 框架：** FastAPI (用于构建 RESTful API 服务，实现 Agent 的部署)
* **向量数据库：** FAISS (用于知识库的向量存储和高效检索)
* **文本嵌入：** OpenAI Embeddings (用于将文本转化为向量)
* **文档加载：** PyPDFLoader (用于加载 PDF 格式的知识文档)
* **依赖管理：** `requirements.txt` 和 `python-dotenv` (用于环境变量管理)
* **版本控制：** Git / GitHub
* **数据处理：** Pandas (在数据分析工具中用于数据操作和统计)
* **网络请求：** Requests (用于 API 工具调用外部服务)

### 2.2 Agent 使用的 Tools 简介

我们的 Agent 具备以下核心工具，能够响应不同的用户需求：

1.  **知识库工具 (`KnowledgeBaseQueryTool`)**
    * **功能：** 允许 Agent 查询预加载的 PDF 文档（例如项目文档、公司手册等）以获取特定信息。
    * **实现：** 利用 `PyPDFLoader` 加载 PDF 文档，`RecursiveCharacterTextSplitter` 进行文本分块，`OpenAIEmbeddings` 生成文本嵌入，`FAISS` 作为向量存储和检索器，最后通过 `RetrievalQA` Chain 实现问答功能。
    * **应用场景：** 用户询问“我的项目文档中关于 LangChain 的 Agent 有什么介绍？”等。

2.  **API 工具 (`WeatherQueryTool`)**
    * **功能：** 调用 OpenWeatherMap 公开 API 查询指定城市的实时天气信息。
    * **实现：** 使用 Python 的 `requests` 库发送 HTTP 请求到 OpenWeatherMap API，解析 JSON 响应，并提取关键天气数据。
    * **应用场景：** 用户询问“上海今天天气怎么样？”、“北京下周的气温预测？”等。

3.  **本地函数工具 (`DataAnalysisAndSortTool`)**
    * **功能：** 对用户提供的一系列数字进行排序和基本的统计分析。
    * **实现：** 这是一个自定义的 Python 函数，接收逗号分隔的数字字符串和排序顺序（升序/降序），利用 Python 内置的 `sorted()` 函数和 Pandas 库进行数据处理和统计描述。
    * **应用场景：** 用户要求“请帮我分析和排序数据：10,5,20,15,8”、“计算这组数据的平均值和中位数”等。

## 3. 运行测试方法说明

本 Agent 系统通过 FastAPI 暴露 RESTful API 接口，用户可以通过发送 HTTP POST 请求来与 Agent 进行对话。

### 3.1 环境准备

1.  **克隆代码库：**
    ```bash
    git clone [https://github.com/](https://github.com/)[你的GitHub用户名]/collaborative-agent-system.git
    cd collaborative_agent_system
    ```
    *请将 `[你的GitHub用户名]` 替换为实际的 GitHub 用户名。*

2.  **创建并激活虚拟环境：**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **安装依赖：**
    ```bash
    pip install -r requirements.txt
    ```
    *如果 `faiss-cpu` 或 `pypdf` 安装遇到问题，请单独尝试安装：`pip install faiss-cpu` 和 `pip install pypdf`。*

4.  **配置 API 密钥：**
    在项目根目录创建 `.env` 文件，并填入你的 OpenAI 和 OpenWeatherMap API 密钥：
    ```env
    OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY_HERE"
    OPENWEATHERMAP_API_KEY="YOUR_OPENWEATHERMAP_API_KEY_HERE"
    ```
    * **重要：** 替换为你的真实 API 密钥。这些密钥不会被上传到版本控制中。

5.  **放置知识库文档：**
    在 `docs/` 目录下放置你的 PDF 文档，并命名为 `your_project_document.pdf`。
    * **例如：** `collaborative_agent_system/docs/your_project_document.pdf`
    * 如果该文件不存在或不是可搜索文本，知识库工具将无法正常工作。

### 3.2 启动服务器

在项目根目录的终端中运行以下命令：
```bash
python app.py
````

成功启动后，你将看到类似 `INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)` 的输出。请保持此终端窗口打开，让服务器持续运行。

### 3.3 测试 Agent

另开一个**新的命令行终端窗口**（在 PyCharm 中可以在“Terminal”选项卡点击 `+` 号），并确保也导航到项目根目录并激活虚拟环境。

你可以使用 `curl` 命令（命令行工具）或 Postman/Insomnia 等 API 测试工具来发送请求。

**API 接口：** `POST http://localhost:8000/chat`
**请求体 (JSON):**

```json
{
    "query": "你的问题或请求"
}
```

**示例测试命令 (请选择适用于你的终端的命令，推荐 PowerShell 使用 `curl.exe` 或 `Invoke-RestMethod`):**

1.  **查询天气：**

      * **通用 (Bash/Zsh/Git Bash):**
        ```bash
        curl -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"query": "上海今天天气怎么样？"}'
        ```
      * **Windows CMD (单行，需要转义双引号):**
        ```cmd
        curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\": \"上海今天天气怎么样？\"}"
        ```
      * **Windows PowerShell (推荐，使用 `curl.exe` 或 `Invoke-RestMethod`):**
        ```powershell
        curl.exe -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\": \"上海今天天气怎么样？\"}"
        # 或者
        Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post -ContentType "application/json" -Body '{"query": "上海今天天气怎么样？"}'
        ```

2.  **数据分析：**

      * **通用 (Bash/Zsh/Git Bash):**
        ```bash
        curl -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"query": "请帮我分析并降序排列数据：5, 12, 3, 8, 1"}'
        ```
      * **Windows CMD (单行，需要转义双引号):**
        ```cmd
        curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\": \"请帮我分析并降序排列数据：5, 12, 3, 8, 1\"}"
        ```
      * **Windows PowerShell:**
        ```powershell
        curl.exe -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\": \"请帮我分析并降序排列数据：5, 12, 3, 8, 1\"}"
        # 或者
        Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post -ContentType "application/json" -Body '{"query": "请帮我分析并降序排列数据：5, 12, 3, 8, 1"}'
        ```

3.  **知识库查询：**

      * **通用 (Bash/Zsh/Git Bash):**
        ```bash
        curl -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"query": "我的项目文档中关于LangChain的Agent有什么介绍？"}'
        ```
      * **Windows CMD (单行，需要转义双引号):**
        ```cmd
        curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\": \"我的项目文档中关于LangChain的Agent有什么介绍？\"}"
        ```
      * **Windows PowerShell:**
        ```powershell
        curl.exe -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\": \"我的项目文档中关于LangChain的Agent有什么介绍？\"}"
        # 或者
        Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post -ContentType "application/json" -Body '{"query": "我的项目文档中关于LangChain的Agent有什么介绍？"}'
        ```

4.  **普通对话：**

      * **通用 (Bash/Zsh/Git Bash):**
        ```bash
        curl -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"query": "你好，你是谁？"}'
        ```
      * **Windows CMD (单行，需要转义双引号):**
        ```cmd
        curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\": \"你好，你是谁？\"}"
        ```
      * **Windows PowerShell:**
        ```powershell
        curl.exe -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\": \"你好，你是谁？\"}"
        # 或者
        Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post -ContentType "application/json" -Body '{"query": "你好，你是谁？"}'
        ```

**观察与调试：**
在执行 `curl` 命令时，请密切关注运行 `app.py` 的那个终端窗口。Agent 的思考过程（`Thought`, `Action`, `Observation` 等）会详细打印在那里，这对于理解 Agent 如何决策和排除故障非常有用。

## 4\. 聊天记录示例

以下是 Agent 与用户进行交互的几个示例，展示了 Agent 如何利用其工具和记忆来辅助用户。

### 示例 1：天气查询

  * **用户输入:** `北京今天天气怎么样？`
  * **Agent 思考 (服务器终端 verbose output 截取):**
    ```
    > Entering AgentExecutor chain...
    Thought: 用户询问北京的天气，我需要使用 WeatherQueryTool 来获取实时天气信息。WeatherQueryTool 的输入是城市的名称。
    Action: WeatherQueryTool
    Action Input: Beijing
    Observation: Location: Beijing, CN
    Temperature: 28.5°C
    Feels Like: 28.7°C
    Humidity: 60%
    Weather: few clouds
    Wind Speed: 3.6 m/s
    Thought: 我已经获取了北京的天气信息。现在我应该将这些信息汇总并以友好的方式返回给用户。
    ...
    > Finished chain.
    ```
  * **Agent 回复 (curl 命令返回的 JSON 响应):**
    ```json
    {
      "response": "北京今天气温28.5°C，体感温度28.7°C，湿度60%，少云，风速3.6m/s。"
    }
    ```
  * **分析此 Agent 如何帮助用户：** 用户无需手动去查询天气网站或应用，Agent 通过调用工具直接提供了所需信息，简化了信息获取流程。

### 示例 2：数据分析

  * **用户输入:** `请帮我分析并升序排列数据：22, 11, 33, 44, 55, 66`
  * **Agent 思考 (服务器终端 verbose output 截取):**
    ```
    > Entering AgentExecutor chain...
    Thought: 用户希望我分析并升序排列一组数据。DataAnalysisAndSortTool 可以完成这个任务，它的输入是逗号分隔的数字字符串和排序顺序。
    Action: DataAnalysisAndSortTool
    Action Input: {"data_str": "22, 11, 33, 44, 55, 66", "sort_order": "ascending"}
    Observation: 原始数据: 22, 11, 33, 44, 55, 66
    排序方式: ascending
    排序后数据: [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
    统计分析:
               values
    count    6.000000
    mean    38.500000
    std     20.206434
    min     11.000000
    25%     24.750000
    50%     38.500000
    75%     52.250000
    max     66.000000
    Thought: 我已经成功地对数据进行了排序和分析。现在我应该将结果清晰地呈现给用户。
    ...
    > Finished chain.
    ```
  * **Agent 回复 (curl 命令返回的 JSON 响应):**
    ```json
    {
      "response": "好的，这是您提供的数据的升序排列和统计分析：\n排序后数据：[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]\n统计分析：\n               values\ncount    6.000000\nmean    38.500000\nstd     20.206434\nmin     11.000000\n25%     24.750000\n50%     38.500000\n75%     52.250000\nmax     66.000000"
    }
    ```
  * **分析此 Agent 如何帮助用户：** 用户可以直接通过对话完成数据处理和初步分析，而无需编写代码或使用专业工具，极大地提高了效率和便捷性。

*(请在这里插入更多实际测试的聊天记录截图或文字记录，以充分展示 Agent 的功能和多轮对话能力。例如，可以添加知识库查询和多轮对话的示例。)*

## 5\. 合作与反思

本次项目由 **牛正武** 和 **孙世纪** 共同完成。

### 5.1 牛正武的贡献与反思

  * **负责内容：**
      * 例如：主要负责 Agent 的整体架构设计、OpenAI LLM 的集成、FastAPI 服务器的搭建和部署。
      * 具体实现了知识库工具的 PDF 加载、文本分割和 FAISS 向量检索部分。
      * 撰写了 `README.md` 报告的框架和部分内容。
      * 在 GitHub 上创建了代码库并进行了初始提交。
  * **学到的内容：**
      * 深入理解了 LangChain 中 Agent、Tool、Memory 的工作原理及其在构建复杂应用中的作用。
      * 掌握了使用 FastAPI 部署 AI 应用的基本流程和注意事项。
      * 学会了如何配置和管理环境变量，提高了代码的安全性。
      * 了解了 Git 和 GitHub 在团队协作中的重要性，以及基本的代码版本管理操作。
  * **遇到的困难：**
      * **困难 1：** 在 Agent 调试阶段，有时 Agent 会出现“幻觉”或者无法正确选择工具。
          * **解决方案：** 尝试调整 LLM 的 `temperature` 参数，并优化 Agent 的 `PromptTemplate`，使其更明确地理解工具的使用场景和输入输出格式。同时，开启 `verbose=True` 观察 Agent 的思维链，有助于定位问题。
      * **困难 2：** 知识库工具在处理某些复杂 PDF 文档时，文本分割效果不理想，导致检索不准确。
          * **解决方案：** 尝试调整 `RecursiveCharacterTextSplitter` 的 `chunk_size` 和 `chunk_overlap` 参数，并考虑使用不同的文本分割策略（如根据标题、段落等）。
      * **困难 3：** 在服务器部署时，API 密钥的加载和管理初期存在问题。
          * **解决方案：** 引入 `python-dotenv` 库，并严格按照 `.env` 文件规范管理 API 密钥，确保其不被硬编码到代码中，同时也在程序启动时进行必要的密钥存在性检查。

### 5.2 孙世纪 的贡献与反思

  * **负责内容：**
      * 例如：主要负责工具的开发，具体实现了天气查询工具和数据分析本地函数工具。
      * 协助测试和调试 Agent 的行为，特别是工具调用的准确性。
      * 参与了 `README.md` 报告的撰写和内容补充。
      * 积极参与了 Git 提交和代码同步，确保了代码库的及时更新。
  * **学到的内容：**
      * 学习了如何将外部 API (如 OpenWeatherMap) 集成到 Agent 中，并处理其响应。
      * 掌握了自定义 Python 函数作为 LangChain 工具的方法，拓展了 Agent 的能力。
      * 对 LangChain 的 `AgentExecutor` 和 `Tool` 接口有了更深的理解。
      * 通过实际操作，巩固了 Git 的常用命令，如 `pull`, `add`, `commit`, `push` 等。
  * **遇到的困难：**
      * **困难 1：** 天气 API 返回的 JSON 格式有时比较复杂，难以准确提取所需信息。
          * **解决方案：** 仔细阅读 API 文档，使用 `json.loads()` 后，逐层解析字典和列表，并通过 `data.get()` 方法安全地访问键，避免因键不存在而引发错误。
      * **困难 2：** 本地函数工具在处理用户输入时，类型转换（如字符串转数字）和错误校验不够完善。
          * **解决方案：** 增加了 `try-except` 块来捕获 `ValueError`，并添加了对空输入和无效排序顺序的校验，使工具更加健壮。
      * **困难 3：** 在与同学协作时，偶尔会遇到代码合并冲突。
          * **解决方案：** 及时进行 `git pull` 同步最新代码，并在每次修改前先拉取，如果出现冲突，则仔细查看冲突部分并手动解决。通过多沟通，避免在同一文件相同区域进行大量修改。

<!-- end list -->

```
```
