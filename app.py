import pprint
from flask import Flask, Response, request, jsonify, session
from flask_session import Session
from flask_cors import CORS
from langchain_ollama import ChatOllama
from typing import Any, Dict, List
from sqlalchemy import create_engine
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
import pandas as pd
import re
from urllib.parse import quote
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.embeddings import OllamaEmbeddings
import json
import random
import string
from datetime import datetime
import redis
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from flask import stream_with_context
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langgraph.prebuilt import create_react_agent
import matplotlib.pyplot as plt
import plotly.express as px
from langchain_core.tools import tool
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain_core.messages import HumanMessage


from config import (
    AUTO_CHART_PROMPT,
    PROMPT_CHART,
    SYSTEM_PREFIX,
    SYSTEM_PREFIX_SQL_TOOL,
    SYSTEM_PREFIX_TOOL,
)

# Flask app
app = Flask(__name__)
app.secret_key = "my_secret_key"

CORS(app, origins=["http://localhost:3000","http://100.83.49.115:3000"], supports_credentials=True)

# Configure Flask-Session to use server-side sessions
app.config["SESSION_TYPE"] = "redis"
app.config["SESSION_PERMANENT"] = False  # Session won't expire unless manually set
app.config["SESSION_USE_SIGNER"] = True  # Adds a secure signature to the session cookie
app.config["SESSION_REDIS"] = redis.StrictRedis(host="localhost", port=6379, db=0)
REDIS_URL = "redis://:password@c0af-2400-9800-2e0-4c58-5081-3c83-8164-9dd2.ngrok-free.app:6379/0"

# app.config["SESSION_REDIS"] = redis.StrictRedis.from_url(REDIS_URL)
# Configure session cookie options

# Initialize the session extension
Session(app)


def get_llm():
    host = session.get("host", "")
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm_model = ChatOllama(
        model="llama3.2",
        temperature=0,
        base_url=host,
        streaming=True,
        callbacks=callback_manager,
    )

    return llm_model


def get_engine_database(guid):

    list_sql_connection = session.get("sql_connection", [])
    database_connection = next(
        (prompt for prompt in list_sql_connection if prompt["guid"] == guid), ""
    )

    encoded_password = quote(database_connection["password"])

    # URL koneksi PostgreSQL Anda
    database_url_encoded = f"{database_connection["database_type"]}://{database_connection["user"]}:{encoded_password}@{database_connection["host"]}:{database_connection["port"]}/{database_connection["database_name"]}"

    engine = create_engine(database_url_encoded)
    return engine


def get_full_prompt(guid: str):
    host = session.get("host", "")
    training_connections = session.get("training_connection", "")

    filtered_training_list = [
        item for item in training_connections if item["database_guid"] == guid
    ]

    training_connection_results = []
    for item in filtered_training_list:
        training_connection_results.append(
            {
                "input": item["input"],
                "query": item["query"],
                "description": item["description"],
            }
        )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        training_connection_results,
        OllamaEmbeddings(model="nomic-embed-text", base_url=host),
        FAISS,
        k=5,
        input_keys=["input"],
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template("{query}"),
        input_variables=["input"],
        prefix=SYSTEM_PREFIX,
        suffix="",
    )

    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
        ]
    )

    return full_prompt

def get_full_prompt_tool(guid: str):
    host = session.get("host", "")
    training_connections = session.get("training_connection", "")

    filtered_training_list = [
        item for item in training_connections if item["database_guid"] == guid
    ]

    training_connection_results = []
    for item in filtered_training_list:
        training_connection_results.append(
            {
                "input": item["input"],
                "query": item["query"],
                "description": item["description"],
            }
        )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        training_connection_results,
        OllamaEmbeddings(model="nomic-embed-text", base_url=host),
        FAISS,
        k=5,
        input_keys=["input"],
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template("{query}"),
        input_variables=["input"],
        prefix=SYSTEM_PREFIX,
        suffix="",
    )

    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
        ]
    )

    return full_prompt


def get_full_prompt_sql_tool(guid: str):
    host = session.get("host", "")
    training_connections = session.get("training_connection", "")

    filtered_training_list = [
        item for item in training_connections if item["database_guid"] == guid
    ]

    training_connection_results = []
    for item in filtered_training_list:
        training_connection_results.append(
            {
                "input": item["input"],
                "query": item["query"],
                "description": item["description"],
            }
        )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        training_connection_results,
        OllamaEmbeddings(model="nomic-embed-text", base_url=host),
        FAISS,
        k=5,
        input_keys=["input"],
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template("{query}"),
        input_variables=["input"],
        prefix=SYSTEM_PREFIX_SQL_TOOL,
        suffix="",
    )

    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    return full_prompt


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


examples = [
    {"input": "List all artists.", "query": "SELECT * FROM Artist;", "description": ""},
    {"input": "List all cms.", "query": "SELECT * FROM cms;", "description": ""},
    {
        "input": "List all transactions.",
        "query": "SELECT * FROM transaction_schema.transactions limit 10;",
        "description": "",
    },
    {
        "input": "List all sidebar menu.",
        "query": "SELECT * FROM sidebar_menu;",
        "description": "",
    },
    {
        "input": "List all last month.",
        "query": "SELECT * FROM last_month;",
        "description": "",
    },
    {
        "input": "Top Product Therapist.",
        "query": "SELECT report_schema.report_top_product_therapist('5cec917d-2aab-48bb-923b-8642a8a3c1cc', false, null , null);",
        "description": "",
    },
]


layout_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            AUTO_CHART_PROMPT,
        ),
        ("human", "{input}"),
    ]
)

query_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PREFIX,
        ),
        ("human", "{input}"),
    ]
)

custom_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
         Jawab langsung ke intinya, jawab dengan bahasa user dan jawab sesuai instruksi: 
         {custom_prompt}""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chart_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            PROMPT_CHART,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

@tool
def execute_sql(sql_query):
    """Execute valid sql query or return final result"""
    # Fungsi ini berfungsi sebagai tool kedua.
    print(f"ini query nya {sql_query}")
    if(is_sql_query(sql_query)):
        engine = get_engine_database("QuVaFRRQlo")
        try:
            data = pd.read_sql(
                    sql_query, engine
                    )  # Execute the SQL query against the database
            df = pd.DataFrame(data)
 
            parser = PandasDataFrameOutputParser(dataframe=df)

            result = str(df.head(len(df)).to_markdown(index=False, floatfmt=".1f"))
            return parser
        except Exception as e:
            return "Just show final result without additional explanations or comments"

    return "Just show final result"

from langchain.agents import tool, initialize_agent, AgentType

@tool
def get_similar_query(question: str) -> str:
    """Dynamically generate SQL queries or responses based on user input."""
    # Convert the input to lowercase to make it case-insensitive
    question = question.lower().strip()
    
    # Basic patterns for dynamic handling
    if "list" in question and "cms" in question:
        return "SELECT * FROM cms"
    
    # Handle variations of general queries
    elif "list" in question:
        # A general query that can apply to any entity
        entity = question.replace("list", "").strip()
        if entity:
            return f"SELECT * FROM {entity}"
        else:
            return "Please specify what you want to list, e.g., 'list users', 'list cms'."
    
    elif "hello" in question or "hi" in question:
        return "Hello, how can I assist you today?"
    
    elif "create" in question and "cms" in question:
        return "INSERT INTO cms (column1, column2, ...) VALUES (value1, value2, ...)"
    
    # A catch-all for other common queries
    elif "show" in question:
        # 'show' could be a command to list tables or schemas
        if "tables" in question:
            return "SHOW TABLES"
        else:
            return "Sorry, I didn't fully understand that. Could you be more specific?"
    
    # Handle "select" queries dynamically
    elif "select" in question:
        return "Please provide the table name and columns you want to select."
    
    # Default case if no recognizable pattern is matched
    else:
        return "Sorry, I didn't understand your query. Could you rephrase it?"


def clean_generation_result(result: str) -> str:
    def _normalize_whitespace(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    return (
        _normalize_whitespace(result)
        .replace("\\n", " ")
        .replace("```sql", "")
        .replace('"""', "")
        .replace("'''", "")
        .replace("```", "")
        .replace(";", "")
    )

def clean_generation_result_stream(stream):
    def _normalize_whitespace(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def validate_chunk(chunk: str) -> bool:
        # Example validation: Ensure the chunk is a non-empty string
        return isinstance(chunk, str) and len(chunk.strip()) > 0

    cleaned_chunks = []
    for chunk in stream:
        print(f"Chunk received: {chunk}")  # Debug log
        if validate_chunk(chunk):
            cleaned_chunk = (
                _normalize_whitespace(chunk)
                .replace("\\n", " ")
                .replace("```sql", "")
                .replace('"""', "")
                .replace("'''", "")
                .replace("```", "")
                .replace(";", "")
            )
            cleaned_chunks.append(cleaned_chunk)

    # Assemble cleaned chunks into a single statement
    return " ".join(cleaned_chunks)


def clean_generation_python(result: str) -> str:
    return result.replace("```python", "").replace("```", "")


# Solely for documentation purposes.
def format_parser_output(parser_output: Dict[str, Any]) -> None:
    for key in parser_output.keys():
        parser_output[key] = parser_output[key].to_dict()
    return pprint.PrettyPrinter(width=4, compact=True).pprint(parser_output)

def generateQuery(input, guid):

    # try:
        llm = get_llm()
        full_prompt = get_full_prompt(guid)
        full_prompt_tool = get_full_prompt_tool(guid)

        tools = [execute_sql]

        llm_tool = llm.bind_tools(tools)

        chain = full_prompt | llm
        chain_tool = full_prompt_tool | llm_tool
        result = chain.invoke({"input": input})
        result_tool = chain_tool.invoke({"input": input})
        sql = clean_generation_result(result.content)
        # agent_executor = create_react_agent(llm, tools)
        # response = agent_executor.invoke({"messages": [HumanMessage(content="SELECT * FROM cms LIMIT 5;"+f" guid_engine={guid}")]})
        agent = initialize_agent(
    tools, 
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    llm=llm, 
    verbose=True,
    prompt=full_prompt_tool
    )
        result_agent = agent.invoke(input)

        print(result_agent)

        if not guid:
            return (
                jsonify({"result": sql}),
                200,
            )

        engine = get_engine_database(guid)

        if is_sql_query(sql):
            try:
                session["current_query"] = sql
                data = pd.read_sql(
                    sql, engine
                )  # Execute the SQL query against the database
                df = pd.DataFrame(data)

                result = str(df.head(len(df)).to_markdown(index=False, floatfmt=".1f"))

                # parsed_result_chart = json.loads(result_chart.to_json())

                # print(json.loads(df.head(len(df)).to_json()))

                return (
                    jsonify({"result": result, "table": True}),
                    200,
                )
            except Exception as e:
                return jsonify({"error": f"An error occurred: {e}"}), 500
        else:
            return jsonify({"result": sql, "table": False}), 200
    # except Exception as e:
    #     return jsonify({"error": "Failed to connect !"}), 500
    
def generateQueryStream(input):

    llm = get_llm()

    chain = query_prompt | llm
    result = chain.invoke({"input": input})

    result_generator = chain.stream(
        {"input": result.content},
    )
    
    for chunk in result_generator:
        yield chunk.content    


def generateChartFromDataframe(df: pd.DataFrame):

    exclude_columns = ["id"]

    # Filter columns to exclude the ones we don't want to use
    df_filtered = df.loc[:, ~df.columns.isin(exclude_columns)]

    # Check for date columns and numeric columns
    date_columns = df_filtered.select_dtypes(
        include=["datetime64"]
    ).columns  # Date columns
    numeric_columns = df_filtered.select_dtypes(
        include=["number"]
    ).columns  # Numeric columns

    if len(date_columns) > 0:
        for col in date_columns:
            # Format dates into 'DD-MM-YYYY' or any desired format
            df_filtered[col] = pd.to_datetime(df_filtered[col]).dt.strftime("%d-%m-%Y")
            # print(f"Formatted date columns: {date_columns.tolist()}")

    # Select x and y columns dynamically
    if len(date_columns) > 0 and len(numeric_columns) > 0:
        # Use the first date column for x and the first numeric column for y
        x_column = date_columns[0]
        y_column = numeric_columns[0]
    elif len(date_columns) > 0:
        # If only date columns are available, use the first one for x
        x_column = date_columns[0]
        y_column = (
            numeric_columns[0] if len(numeric_columns) > 0 else df_filtered.columns[0]
        )
    elif len(numeric_columns) > 0:
        # If only numeric columns are available, use the first two for x and y
        x_column = numeric_columns[0]
        y_column = (
            numeric_columns[1] if len(numeric_columns) > 1 else df_filtered.columns[0]
        )
    else:
        print("Not enough numeric or date data for plotting.")
        x_column = df_filtered.columns[0]  # Default fallback
        y_column = df_filtered.columns[0]  # Default fallback

    fig = px.bar(df_filtered, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    return fig


def generateJson(input):

    sql_query = session.get("current_query", "")
    print(sql_query)
    if is_sql_query(sql_query):
        data = pd.read_sql(
            sql_query, engine
        )  # Execute the SQL query against the database
        df = pd.DataFrame(data)

    else:
        df = pd.DataFrame([sql_query])

    chain_layout = layout_prompt | llm
    result = chain_layout.invoke(
        {"input": input, "json_input": df.head().to_json(orient="split")}
    )
    sql = result.content
    return sql


def generateFromCustomPrompt(input, custom_prompt_text):

    llm = get_llm()

    chain_custom_prompt = custom_prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain_custom_prompt,
        get_by_session_id,  # Function to retrieve session history
        input_messages_key="input",  # Key for the input messages
        history_messages_key="history",  # Key for the message history
    )

    result = chain_with_history.invoke(  # noqa: T201
        {"input": input, "custom_prompt": custom_prompt_text},
        config={"configurable": {"session_id": "custom_prompt"}},
    )
    # result = chain_custom_prompt.invoke(
    #     {"input": input, "custom_prompt": custom_prompt_text}
    # )
    return result.content


def generateFromChart(input, guid):

    llm = get_llm()

    chain_chart = chart_prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain_chart,
        get_by_session_id,  # Function to retrieve session history
        input_messages_key="input",  # Key for the input messages
        history_messages_key="history",  # Key for the message history
    )

    sql_query = session.get("current_query", "")
    engine = get_engine_database(guid)
    data = pd.read_sql(sql_query, engine)  # Execute the SQL query against the database
    df = pd.DataFrame(data)

    column_info = "\n".join(
        [f"- {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)]
    )

    result = chain_with_history.invoke(  # noqa: T201
        {
            "input": input,
            "column_info": column_info,
        },
        config={"configurable": {"session_id": "chart"}}
    )

    tool = PythonAstREPLTool(locals={"df": df, "px": px})

    fixed_code = clean_generation_python(result.content)
    output = tool.run(fixed_code)
    print(output)
    return json.loads(output)


def generateFromCustomPromptStream(input, custom_prompt_text):
    llm = get_llm()

    chain_custom_prompt = custom_prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain_custom_prompt,
        get_by_session_id,  # Function to retrieve session history
        input_messages_key="input",  # Key for the input messages
        history_messages_key="history",  # Key for the message history
    )

    # Simulate streaming results from the LLM (assuming the LLM supports this)
    result_generator = chain_with_history.stream(
        {"input": input, "custom_prompt": custom_prompt_text},
        config={
            "configurable": {
                "session_id": (
                    "custom_prompt" if not custom_prompt_text else "default_prompt"
                )
            }
        },
    )

    for chunk in result_generator:
        yield chunk.content


def is_sql_query(query):
    sql_keywords = [
        "SELECT",
        "FROM",
        "WHERE",
        "JOIN",
        "GROUP BY",
        "ORDER BY",
        "LIMIT",
        "VALUES",
        "INTO",
    ]
    query = query.upper()
    return any(
        re.search(r"\b" + re.escape(keyword) + r"\b", query) for keyword in sql_keywords
    )


# Function to generate a random string
def generate_random_string(length):
    # Choose random characters from letters and digits
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


# Flask endpoint for generateQuery
@app.route("/generate-query", methods=["POST"])
def handle_generate_query():
    data = request.json
    input_query = data.get("input")
    guid = data.get("guid")

    if not input_query or not guid:
        return jsonify({"error": "Missing input query"}), 400

    sql = generateQuery(input_query, guid)

    # session["current_query"] = sql

    return sql


@app.route("/nl-to-sql-stream", methods=["POST"])
def handle_nl_to_sql():
    # Get the JSON payload from the request
    data = request.json
    input_query = data.get("input")  # Extract the 'input' field

    # Check if both 'input' and 'sql_input' are provided
    if not input_query:
        return jsonify({"error": "Missing input query or SQL query"}), 400

    def generate_response():
        for chunk in generateQueryStream(
            input_query
        ):
            yield chunk  # Optional: separate chunks with newlines for easier parsing

    return Response(stream_with_context(generate_response()), content_type="text/plain")


@app.route("/generate-from-custom-prompt", methods=["POST"])
def handle_generate_from_custom_prompt():
    # Get the JSON payload from the request
    data = request.json
    input_query = data.get("input")  # Extract the 'input' field
    guid = data.get("guid")  # Extract the 'input' field

    # Check if both 'input' and 'sql_input' are provided
    if not input_query:
        return jsonify({"error": "Missing input or custom prompt"}), 400

    prompts = session.get("custom_prompt", [])

    custom_prompt = next((prompt for prompt in prompts if prompt["guid"] == guid), "")

    # Assuming you have a function 'generateJson' to process the queries
    result = generateFromCustomPrompt(input_query, custom_prompt_text=custom_prompt)
    # Return the result from generateJson in the response
    return jsonify({"result": result}), 200


@app.route("/generate-from-custom-prompt-stream", methods=["POST"])
def handle_generate_from_custom_prompt_stream():
    data = request.json
    input_query = data.get("input")
    guid = data.get("guid")

    if not input_query:
        return jsonify({"error": "Missing input or custom prompt"}), 400

    prompts = session.get("custom_prompt", [])
    custom_prompt = next((prompt for prompt in prompts if prompt["guid"] == guid), "")

    def generate_response():
        for chunk in generateFromCustomPromptStream(
            input_query, custom_prompt_text=custom_prompt
        ):
            yield chunk  # Optional: separate chunks with newlines for easier parsing

    return Response(stream_with_context(generate_response()), content_type="text/plain")

@app.route("/generate-from-chart", methods=["POST"])
def handle_generate_from_chart():
    # Get the JSON payload from the request
    data = request.json
    input_query = data.get("input")  # Extract the 'input' field
    guid = data.get("guid")  # Extract the 'input' field

    if not input_query:
        return jsonify({"error": "Missing input"}), 400

    result = generateFromChart(input_query, guid)

    return jsonify({"result": result}), 200

@app.route("/custom-prompt", methods=["POST"])
def handle_create_custom_prompt():
    # Get the JSON payload from the request
    data = request.json
    name = data.get("name")  # Extract the 'input' field
    custom_prompt = data.get("custom_prompt")  # Extract the 'input' field

    # Check if both 'input' and 'sql_input' are provided
    if not name or not custom_prompt:
        return jsonify({"error": "Missing input or custom prompt"}), 400

    prompts = session.get("custom_prompt", [])

    now = datetime.now()

    # Convert to string with a literal 'Z' at the end
    date_string = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    data["guid"] = generate_random_string(10)
    data["created_at"] = date_string

    # Add the new prompt to the list
    prompts.append(data)

    session["custom_prompt"] = prompts

    # Return the result from generateJson in the response
    return jsonify({"result": "Success"}), 200


@app.route("/custom-prompt", methods=["PUT"])
def handle_update_custom_prompt():
    # Get the JSON payload from the request
    data = request.json
    guid = data.get("guid")
    name = data.get("name")
    custom_prompt = data.get("custom_prompt")

    # Check if 'guid' is provided
    if not guid:
        return jsonify({"error": "Missing 'guid' for update"}), 400

    prompts = session.get("custom_prompt", [])

    # Find the prompt with the given GUID
    prompt = next((prompt for prompt in prompts if prompt["guid"] == guid), None)

    if not prompt:
        return jsonify({"error": "Prompt not found"}), 404

    # Update fields if provided
    if name:
        prompt["name"] = name
    if custom_prompt:
        prompt["custom_prompt"] = custom_prompt

    session["custom_prompt"] = prompts
    return jsonify({"result": "Prompt updated successfully"}), 200


@app.route("/custom-prompt/<guid>", methods=["DELETE"])
def handle_delete_custom_prompt(guid):
    # Check if 'guid' is provided in the URL
    if not guid:
        return jsonify({"error": "Missing 'guid' in URL"}), 400

    prompts = session.get("custom_prompt", [])

    # Filter out the prompt with the given GUID
    updated_prompts = [prompt for prompt in prompts if prompt["guid"] != guid]

    if len(prompts) == len(updated_prompts):  # No change means the guid was not found
        return jsonify({"error": "Prompt not found"}), 404

    # Update session
    session["custom_prompt"] = updated_prompts

    return jsonify({"result": "Prompt deleted successfully"}), 200


@app.route("/set-ai-host", methods=["POST"])
def handle_set_ai_host():
    # Get the JSON payload from the request
    data = request.json
    host = data.get("host")  # Extract the 'input' field

    # Check if both 'input' and 'sql_input' are provided
    if not host:
        return jsonify({"error": "Missing host"}), 400

    session["host"] = host

    global llm
    llm = get_llm()

    # Return the result from generateJson in the response
    return jsonify({"result": "Success"}), 200


@app.route("/get-ai-host", methods=["GET"])
def get_ai_host():
    # Remove the JSON list from the session (GET method)
    host = session.get("host", "")
    return jsonify({"result": host}), 200


@app.route("/sql-connection", methods=["POST"])
def handle_create_sql_connection():
    # Get the JSON payload from the request
    data = request.json
    # Check if both 'input' and 'sql_input' are provided
    required_fields = ["name", "database_name", "user", "password", "host", "port"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing field!"}), 400

    prompts = session.get("sql_connection", [])

    now = datetime.now()

    # Convert to string with a literal 'Z' at the end
    date_string = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    data["guid"] = generate_random_string(10)
    data["created_at"] = date_string

    # Add the new prompt to the list
    prompts.append(data)

    session["sql_connection"] = prompts

    # Return the result from generateJson in the response
    return jsonify({"result": "Success"}), 200


@app.route("/sql-connection", methods=["PUT"])
def handle_update_sql_connection():
    # Get the JSON payload from the request
    data = request.json
    guid = data.get("guid")
    name = data.get("name")
    database_name = data.get("database_name")
    database_logo = data.get("database_logo")
    database_type = data.get("database_type")
    user = data.get("user")
    password = data.get("password")
    host = data.get("host")
    port = data.get("port")

    required_fields = ["name", "database_name", "user", "password", "host", "port"]

    # Check if 'guid' is provided
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing 'guid' for update"}), 400

    prompts = session.get("sql_connection", [])

    # Find the prompt with the given GUID
    prompt = next((prompt for prompt in prompts if prompt["guid"] == guid), None)

    if not prompt:
        return jsonify({"error": "Prompt not found"}), 404

    # Update fields if provided
    if name:
        prompt["name"] = name
    if database_name:
        prompt["database_name"] = database_name
    if user:
        prompt["user"] = user
    if password:
        prompt["password"] = password
    if host:
        prompt["host"] = host
    if port:
        prompt["port"] = port
    if database_logo:
        prompt["database_logo"] = database_logo
    if database_type:
        prompt["database_type"] = database_type

    session["sql_connection"] = prompts
    return jsonify({"result": "Prompt updated successfully"}), 200


@app.route("/sql-connection/<guid>", methods=["DELETE"])
def handle_delete_sql_connection(guid):
    # Check if 'guid' is provided in the URL
    if not guid:
        return jsonify({"error": "Missing 'guid' in URL"}), 400

    prompts = session.get("sql_connection", [])

    # Filter out the prompt with the given GUID
    updated_prompts = [prompt for prompt in prompts if prompt["guid"] != guid]

    if len(prompts) == len(updated_prompts):  # No change means the guid was not found
        return jsonify({"error": "Prompt not found"}), 404

    # Update session
    session["sql_connection"] = updated_prompts

    return jsonify({"result": "Prompt deleted successfully"}), 200


@app.route("/training-connection", methods=["POST"])
def handle_create_training_connection():
    # Get the JSON payload from the request
    data = request.json

    # Check if the payload contains a list of objects
    if not isinstance(data, list):
        return jsonify({"error": "Payload must be a list of objects"}), 400

    required_fields = ["database_guid", "input", "query", "description"]

    # Retrieve or initialize the session data
    prompts = []

    new_entries = []
    for entry in data:
        # Check if all required fields are present
        if not all(field in entry for field in required_fields):
            return jsonify({"error": f"Missing field in entry: {entry}"}), 400

        # Generate a GUID and add it to the entry
        entry["guid"] = generate_random_string(10)
        new_entries.append(entry)

    # Add the new entries to the session's prompts
    prompts.extend(new_entries)
    session["training_connection"] = prompts

    # Return success response
    return jsonify({"result": "Success", "added_entries": new_entries}), 200

@app.route("/thread", methods=["POST"])
def handle_create_thread():
    # Get the JSON payload from the request
    data = request.json
    # Check if both 'input' and 'sql_input' are provided
    required_fields = ["name", "source_guid", "type"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing field!"}), 400

    prompts = session.get("sql_connection", [])

    now = datetime.now()

    # Convert to string with a literal 'Z' at the end
    date_string = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    data["guid"] = generate_random_string(10)
    data["created_at"] = date_string

    # Add the new prompt to the list
    prompts.append(data)

    session["thread"] = prompts

    # Return the result from generateJson in the response
    return jsonify({"result": "Success"}), 200


@app.route("/thread", methods=["PUT"])
def handle_update_thread():
    # Get the JSON payload from the request
    data = request.json
    guid = data.get("guid")
    name = data.get("name")
    source_guid = data.get("source_guid")
    type = data.get("type")

    required_fields = ["guid", "name", "source_guid", "type"]

    # Check if 'guid' is provided
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing 'guid' for update"}), 400

    threads = session.get("thread", [])

    # Find the prompt with the given GUID
    thread = next((thread for thread in threads if thread["guid"] == guid), None)

    if not thread:
        return jsonify({"error": "Thread not found"}), 404

    # Update fields if provided
    if name:
        thread["name"] = name
    if source_guid:
        thread["source_guid"] = source_guid
    if type:
        thread["type"] = type

    session["thread"] = threads
    return jsonify({"result": "Prompt updated successfully"}), 200


@app.route("/thread/<guid>", methods=["DELETE"])
def handle_delete_thread(guid):
    # Check if 'guid' is provided in the URL
    if not guid:
        return jsonify({"error": "Missing 'guid' in URL"}), 400

    threads = session.get("thread", [])

    # Filter out the prompt with the given GUID
    updated_threads = [thread for thread in threads if thread["guid"] != guid]

    if len(threads) == len(updated_threads):  # No change means the guid was not found
        return jsonify({"error": "Prompt not found"}), 404

    # Update session
    session["thread"] = updated_threads

    return jsonify({"result": "Prompt deleted successfully"}), 200


@app.route("/get-list-custom-prompt", methods=["GET"])
def handle_get_list_custom_prompt():
    # Remove the JSON list from the session (GET method)
    data_list = session.get("custom_prompt", [])
    return jsonify({"result": data_list}), 200


@app.route("/get-list-sql-connection", methods=["GET"])
def handle_get_list_sql_connection():
    # Remove the JSON list from the session (GET method)
    data_list = session.get("sql_connection", [])
    return jsonify({"result": data_list}), 200


@app.route("/get-list-training-connection/<guid>", methods=["GET"])
def handle_get_list_training_connection(guid):
    # Remove the JSON list from the session (GET method)
    data_list = session.get("training_connection", [])
    filtered_list = [item for item in data_list if item["database_guid"] == guid]
    return jsonify({"result": filtered_list}), 200


@app.route("/get-list-thread", methods=["GET"])
def handle_get_list_thread():
    # Remove the JSON list from the session (GET method)
    data_list = session.get("thread", [])
    return jsonify({"result": data_list}), 200


@app.route("/get-list-source", methods=["GET"])
def handle_get_list_core():
    # Remove the JSON list from the session (GET method)
    list_sql_connection = session.get("sql_connection", [])
    list_custom_prompt = session.get("custom_prompt", [])

    # Transforming data into the desired model
    result = []

    for item in list_custom_prompt:
        result.append(
            {
                "guid": item["guid"],
                "type": "custom_prompt",  # Assuming type1 for items in list1
                "name": item["name"],
            }
        )

    for item in list_sql_connection:
        result.append(
            {
                "guid": item["guid"],
                "type": "database",  # Assuming type2 for items in list2
                "name": item["name"],
            }
        )

    return jsonify({"result": result}), 200


# Run Flask app
if __name__ == "__main__":
    # app.run(port=8000, debug=True)
    app.run(host="0.0.0.0", port=8000, debug=True)
