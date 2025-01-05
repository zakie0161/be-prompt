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
import matplotlib.pyplot as plt
import plotly.express as px
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain_core.messages import HumanMessage
import sqlite3
import os
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.agents import tool

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
# app.config["SESSION_TYPE"] = "redis"
# app.config["SESSION_PERMANENT"] = False  # Session won't expire unless manually set
# app.config["SESSION_USE_SIGNER"] = True  # Adds a secure signature to the session cookie
# app.config["SESSION_REDIS"] = redis.StrictRedis(host="localhost", port=6379, db=0)
# REDIS_URL = "redis://:password@c0af-2400-9800-2e0-4c58-5081-3c83-8164-9dd2.ngrok-free.app:6379/0"

# app.config["SESSION_REDIS"] = redis.StrictRedis.from_url(REDIS_URL)
# Configure session cookie options

# Initialize the session extension
# Session(app)

global_data = {
    "host": "https://goose-helped-marmot.ngrok-free.app",
    "current_query": ""
}

folder_path = "database"
db_name = "ai-db.db"
db_path = os.path.join(folder_path, db_name)

def get_llm():
    host = global_data.get("host", "")
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm_model = ChatOllama(
        model="llama3.2",
        temperature=0,
        base_url=host,
        streaming=True,
        callbacks=callback_manager,
    )

    return llm_model

def get_db_connection():
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def initSqlite():
   
    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    conn = get_db_connection()

    print(f"Database created successfully at {db_path}")

    cursor = conn.cursor()

    # SQL statement to create the "thread_detail" table
    create_table_queries = [
        """
        CREATE TABLE IF NOT EXISTS thread_detail (
            guid TEXT PRIMARY KEY,
            source TEXT,
            type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS thread (
            guid TEXT PRIMARY KEY,
            created_at TEXT,
            name TEXT,
            type TEXT,
            source_guid TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS sql_connection (
            guid TEXT PRIMARY KEY,
            name TEXT,
            database_name TEXT,
            database_logo TEXT,
            database_type TEXT,
            user TEXT,
            password TEXT,
            host TEXT,
            port INTEGER
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS custom_prompt (
            guid TEXT PRIMARY KEY,
            name TEXT,
            custom_prompt TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS training_connection (
            guid TEXT PRIMARY KEY,
            database_guid TEXT NOT NULL,
            input TEXT NOT NULL,
            query TEXT NOT NULL,
            description TEXT
        );
        """
    ]

    # Execute each SQL query
    for query in create_table_queries:
        cursor.execute(query)
        print("Table 'thread_detail' created successfully!")

    # Commit changes and close the connection
    conn.commit()

    # Close the connection
    conn.close()

def get_engine_database(guid):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch the connection details using the provided GUID
    cursor.execute("SELECT * FROM sql_connection WHERE guid = ?", (guid,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise ValueError(f"No database connection found for GUID: {guid}")

    # Map row to dictionary
    columns = ["guid", "name", "database_name", "database_logo", "database_type", "user", "password", "host", "port"]
    database_connection = dict(zip(columns, row))

    encoded_password = quote(database_connection["password"])

    # Create the database URL
    database_url_encoded = (
        f"{database_connection['database_type']}://{database_connection['user']}:{encoded_password}"
        f"@{database_connection['host']}:{database_connection['port']}/{database_connection['database_name']}"
    )

    # Return SQLAlchemy engine
    engine = create_engine(database_url_encoded)
    return engine


def get_full_prompt(guid: str):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch training connections for the given database GUID
    cursor.execute("SELECT input, query, description FROM training_connection WHERE database_guid = ?", (guid,))
    rows = cursor.fetchall()
    conn.close()

    # Convert rows to dictionaries
    training_connection_results = [
        {"input": row[0], "query": row[1], "description": row[2]} for row in rows
    ]

    if not training_connection_results:
        raise ValueError(f"No training connections found for GUID: {guid}")

    # Example selector
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        training_connection_results,
        OllamaEmbeddings(model="nomic-embed-text", base_url=global_data.get("host", "")),
        FAISS,
        k=5,
        input_keys=["input"],
    )

    # Few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template("{query}"),
        input_variables=["input"],
        prefix=SYSTEM_PREFIX,
        suffix="",
    )

    # Full prompt
    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
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

    try:
        llm = get_llm()
        full_prompt = get_full_prompt(guid)

        chain = full_prompt | llm
        result = chain.invoke({"input": input})
        sql = clean_generation_result(result.content)

        if not guid:
            return (
                jsonify({"result": sql}),
                200,
            )

        engine = get_engine_database(guid)

        if is_sql_query(sql):
            try:
                global_data["current_query"] = sql
                data = pd.read_sql(
                    sql, engine
                )  # Execute the SQL query against the database
                df = pd.DataFrame(data)

                result = str(df.head(len(df)).to_markdown(index=False, floatfmt=".1f"))

                return (
                    jsonify({"result": result, "table": True}),
                    200,
                )
            except Exception as e:
                return jsonify({"error": f"An error occurred: {e}"}), 500
        else:
            return jsonify({"result": sql, "table": False}), 200
    except Exception as e:
        return jsonify({"error": "Failed to connect !"}), 500
    
def generateQueryStream(input):

    llm = get_llm()

    chain = query_prompt | llm
    result = chain.invoke({"input": input})

    result_generator = chain.stream(
        {"input": result.content},
    )
    
    for chunk in result_generator:
        yield chunk.content    


def generateFromChart(input, guid):

    llm = get_llm()

    chain_chart = chart_prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain_chart,
        get_by_session_id,  # Function to retrieve session history
        input_messages_key="input",  # Key for the input messages
        history_messages_key="history",  # Key for the message history
    )

    sql_query = global_data.get("current_query", "")
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


def generateFromCustomPromptStream(input, custom_prompt_text, thread_guid):
    llm = get_llm()

    chain_custom_prompt = custom_prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain_custom_prompt,
        # get_by_session_id,  # Function to retrieve session history
        lambda session_id: SQLChatMessageHistory(
            session_id=session_id, connection_string=f"sqlite:///{db_path}"
        ),
        input_messages_key="input",  # Key for the input messages
        history_messages_key="history",  # Key for the message history
    )
    
    # Simulate streaming results from the LLM (assuming the LLM supports this)
    result_generator = chain_with_history.stream(
        {"input": input, "custom_prompt": custom_prompt_text},
        config={
            "configurable": {
                "session_id": thread_guid
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

def fetch_custom_prompt_by_guid(guid):
    """Fetch a custom prompt by GUID from the SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT custom_prompt FROM custom_prompt WHERE guid = ?", (guid,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise ValueError(f"No custom prompt found for GUID: {guid}")
    
    return row[0]  # Return the custom_prompt text

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


@app.route("/generate-from-custom-prompt-stream", methods=["POST"])
def handle_generate_from_custom_prompt_stream():
    data = request.json
    input_query = data.get("input")
    guid = data.get("guid")
    thread_guid = data.get("thread_guid")

    if not input_query:
        return jsonify({"error": "Missing input"}), 400
    
    custom_prompt = ""
    
    if(guid):
        try:
            # Fetch the custom prompt from the SQLite database
            custom_prompt = fetch_custom_prompt_by_guid(guid)
        except ValueError as e:
            return jsonify({"error": str(e)}), 404

    def generate_response():
        # Assuming `generateFromCustomPromptStream` is defined elsewhere
        for chunk in generateFromCustomPromptStream(
            input_query, custom_prompt_text=custom_prompt, thread_guid=thread_guid
        ):
            yield chunk  # Optionally separate chunks with newlines for easier parsing

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
    name = data.get("name")  # Extract the 'name' field
    custom_prompt = data.get("custom_prompt")  # Extract the 'custom_prompt' field

    # Check if both 'name' and 'custom_prompt' are provided
    if not name or not custom_prompt:
        return jsonify({"error": "Missing 'name' or 'custom_prompt'"}), 400

    try:
        # Generate a unique GUID for the new custom prompt
        guid = generate_random_string(10)

        # Connect to the SQLite database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert the new custom prompt into the database
        cursor.execute(
            """
            INSERT INTO custom_prompt (guid, name, custom_prompt)
            VALUES (?, ?, ?)
            """,
            (guid, name, custom_prompt)
        )

        # Commit the transaction and close the connection
        conn.commit()
        conn.close()

        # Return success response
        return jsonify({"result": "Success", "guid": guid}), 200

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


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

    try:
        # Connect to the SQLite database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if the custom prompt exists in the database
        cursor.execute(
            "SELECT * FROM custom_prompt WHERE guid = ?",
            (guid,)
        )
        prompt = cursor.fetchone()

        if not prompt:
            return jsonify({"error": "Custom prompt not found"}), 404

        # Update fields if provided
        if name:
            cursor.execute(
                "UPDATE custom_prompt SET name = ? WHERE guid = ?",
                (name, guid)
            )
        if custom_prompt:
            cursor.execute(
                "UPDATE custom_prompt SET custom_prompt = ? WHERE guid = ?",
                (custom_prompt, guid)
            )

        # Commit the transaction and close the connection
        conn.commit()
        conn.close()

        # Return success response
        return jsonify({"result": "Custom prompt updated successfully"}), 200

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route("/custom-prompt/<guid>", methods=["DELETE"])
def handle_delete_custom_prompt(guid):
    # Check if 'guid' is provided in the URL
    if not guid:
        return jsonify({"error": "Missing 'guid' in URL"}), 400

    try:
        # Connect to the SQLite database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if the custom prompt exists in the database
        cursor.execute(
            "SELECT * FROM custom_prompt WHERE guid = ?",
            (guid,)
        )
        prompt = cursor.fetchone()

        if not prompt:
            return jsonify({"error": "Custom prompt not found"}), 404

        # Delete the custom prompt from the database
        cursor.execute(
            "DELETE FROM custom_prompt WHERE guid = ?",
            (guid,)
        )

        # Commit the transaction and close the connection
        conn.commit()
        conn.close()

        # Return success response
        return jsonify({"result": "Custom prompt deleted successfully"}), 200

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route("/set-ai-host", methods=["POST"])
def handle_set_ai_host():
    # Get the JSON payload from the request
    data = request.json
    host = data.get("host")  # Extract the 'host' field

    # Check if 'host' is provided
    if not host:
        return jsonify({"error": "Missing host"}), 400

    # Set the global host value
    global_data["host"] = host

    global llm
    llm = get_llm()

    # Return the result
    return jsonify({"result": "Success"}), 200


@app.route("/get-ai-host", methods=["GET"])
def get_ai_host():
    # Retrieve the host from the global data
    host = global_data.get("host", "")
    return jsonify({"result": host}), 200


@app.route("/sql-connection", methods=["POST"])
def handle_create_sql_connection():
    data = request.json

    # Required fields
    required_fields = ["name", "database_name", "user", "password", "host", "port"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing field!"}), 400

    guid = generate_random_string(10)
    name = data["name"]
    database_name = data["database_name"]
    database_logo = data.get("database_logo")  # Optional
    database_type = data.get("database_type")  # Optional
    user = data["user"]
    password = data["password"]
    host = data["host"]
    port = data["port"]

    try:
        # Insert into SQLite
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sql_connection (guid, name, database_name, database_logo, database_type, user, password, host, port)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (guid, name, database_name, database_logo, database_type, user, password, host, port)
        )
        conn.commit()
        conn.close()

        return jsonify({"result": "Success", "guid": guid}), 200

    except sqlite3.IntegrityError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route("/sql-connection", methods=["PUT"])
def handle_update_sql_connection():
    data = request.json

    # Validate required fields
    guid = data.get("guid")
    if not guid:
        return jsonify({"error": "Missing 'guid' for update"}), 400

    # Prepare updated values
    name = data.get("name")
    database_name = data.get("database_name")
    database_logo = data.get("database_logo")
    database_type = data.get("database_type")
    user = data.get("user")
    password = data.get("password")
    host = data.get("host")
    port = data.get("port")

    # Build the SQL update query dynamically
    fields = []
    values = []
    if name:
        fields.append("name = ?")
        values.append(name)
    if database_name:
        fields.append("database_name = ?")
        values.append(database_name)
    if database_logo:
        fields.append("database_logo = ?")
        values.append(database_logo)
    if database_type:
        fields.append("database_type = ?")
        values.append(database_type)
    if user:
        fields.append("user = ?")
        values.append(user)
    if password:
        fields.append("password = ?")
        values.append(password)
    if host:
        fields.append("host = ?")
        values.append(host)
    if port:
        fields.append("port = ?")
        values.append(port)

    # Ensure at least one field is being updated
    if not fields:
        return jsonify({"error": "No fields provided for update"}), 400

    values.append(guid)  # Add `guid` to the values list

    try:
        # Update the record
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"""
            UPDATE sql_connection
            SET {', '.join(fields)}
            WHERE guid = ?
            """,
            values
        )
        conn.commit()
        conn.close()

        if cursor.rowcount == 0:  # No rows updated
            return jsonify({"error": "SQL connection not found"}), 404

        return jsonify({"result": "SQL connection updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route("/sql-connection/<guid>", methods=["DELETE"])
def handle_delete_sql_connection(guid):
    if not guid:
        return jsonify({"error": "Missing 'guid' in URL"}), 400

    try:
        # Delete from SQLite
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sql_connection WHERE guid = ?", (guid,))
        conn.commit()
        conn.close()

        if cursor.rowcount == 0:  # No rows deleted
            return jsonify({"error": "SQL connection not found"}), 404

        return jsonify({"result": "SQL connection deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route("/training-connection", methods=["POST"])
def handle_create_training_connection():
    # Get the JSON payload from the request
    data = request.json

    # Ensure the payload is a list of objects
    if not isinstance(data, list):
        return jsonify({"error": "Payload must be a list of objects"}), 400

    required_fields = ["database_guid", "input", "query", "description"]

    new_entries = []

    try:
        # Connect to the SQLite database
        conn = get_db_connection()
        cursor = conn.cursor()

        # **Step 1: Delete all existing data**
        cursor.execute("DELETE FROM training_connection")

        # Prepare a list for batch insertion
        batch_insert_data = []

        for entry in data:
            # Check if all required fields are present
            if not all(field in entry for field in required_fields):
                return jsonify({"error": f"Missing field in entry: {entry}"}), 400

            # Generate a unique GUID for the record
            guid = generate_random_string(10)

            # Prepare data for batch insertion
            batch_insert_data.append((
                guid,
                entry["database_guid"],
                entry["input"],
                entry["query"],
                entry.get("description")
            ))

            # Add the inserted entry (with generated GUID) to the response list
            entry_with_guid = {
                "guid": guid,
                "database_guid": entry["database_guid"],
                "input": entry["input"],
                "query": entry["query"],
                "description": entry.get("description"),
            }
            new_entries.append(entry_with_guid)

        # Perform the batch insert
        cursor.executemany(
            """
            INSERT INTO training_connection (guid, database_guid, input, query, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            batch_insert_data
        )

        # Commit the transaction
        conn.commit()
        conn.close()

        # Return success response with added entries
        return jsonify({"result": "Success", "added_entries": new_entries}), 200

    except sqlite3.IntegrityError as e:
        return jsonify({"error": "Integrity Error", "details": str(e)}), 400
    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500
    

@app.route("/thread", methods=["POST"])
def handle_create_thread():
    # Get the JSON payload from the request
    data = request.json
    # Check if both 'input' and 'sql_input' are provided
    required_fields = ["name"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing field!"}), 400

    now = datetime.now()

    type_ = data.get("type")
    name = data.get("name")
    source_guid = data.get("source_guid")

    # Convert to string with a literal 'Z' at the end
    date_string = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_string_db = now.strftime("%Y-%m-%d %H:%M:%S")
    guid = generate_random_string(10)
    data["guid"] = guid
    data["created_at"] = date_string
    data["name"] = name
    data["type"] = type_
    data["source_guid"] = source_guid

     # Insert into the database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO thread_detail (guid, type, source, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (guid, type_ if type_ else None, source_guid if source_guid else None, date_string_db)  # Make sure you provide 4 values
        )
        cursor.execute(
            """
            INSERT INTO thread (guid, created_at, name, type, source_guid)
            VALUES (?, ?, ?, ?, ?)
            """,
            (guid, date_string_db, name, type_ if type_ else None, source_guid if source_guid else None)
        )
        conn.commit()
        conn.close()

        # Return success response
        return jsonify({"result": "Success", "guid": guid}), 200
    except sqlite3.IntegrityError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route("/thread", methods=["PUT"])
def handle_update_thread():
    data = request.json
    required_fields = ["guid", "name", "source_guid", "type"]

    # Validate required fields
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    guid = data["guid"]
    name = data["name"]
    source_guid = data["source_guid"]
    type_ = data["type"]

    try:
        # Update record in SQLite
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE thread
            SET name = ?, source_guid = ?, type = ?
            WHERE guid = ?
            """,
            (name, source_guid, type_, guid)
        )
        conn.commit()
        conn.close()

        if cursor.rowcount == 0:  # No rows updated
            return jsonify({"error": "Thread not found"}), 404

        return jsonify({"result": "Thread updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route("/thread/<guid>", methods=["DELETE"])
def handle_delete_thread(guid):
    # Check if 'guid' is provided in the URL
    if not guid:
        return jsonify({"error": "Missing 'guid' in URL"}), 400

    try:
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Query the database for the thread details using the given GUID
        cursor.execute("DELETE FROM thread_detail WHERE guid = ?", (guid,))
        cursor.execute("DELETE FROM thread WHERE guid = ?", (guid,))
        
        conn.commit()
        # Close the connection
        conn.close()

        # Return the thread details along with the message history
        return jsonify({"result": "Thread deleted successfully"}), 200

    except Exception as e:
        # Handle any unexpected server errors
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route("/clear-thread", methods=["DELETE"])
def handle_clear_thread():
    try:
        # Connect to the SQLite database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Clear all records from the `thread` table
        cursor.execute("DELETE FROM thread")
        cursor.execute("DELETE FROM thread_detail")
        conn.commit()
        conn.close()

        # Return success response
        return jsonify({"result": "All threads deleted successfully"}), 200

    except Exception as e:
        # Handle any unexpected errors
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route("/get-list-custom-prompt", methods=["GET"])
def handle_get_list_custom_prompt():
    db = get_db_connection()
    cursor = db.cursor()
    
    # Fetch the data from the 'custom_prompt' table
    cursor.execute("SELECT * FROM custom_prompt")
    data_list = cursor.fetchall()
    
    # Convert result to a list of dictionaries
    data_list = [dict(row) for row in data_list]
    
    return jsonify({"result": data_list}), 200

@app.route("/get-list-sql-connection", methods=["GET"])
def handle_get_list_sql_connection():
    db = get_db_connection()
    cursor = db.cursor()
    
    # Fetch the data from the 'sql_connection' table
    cursor.execute("SELECT * FROM sql_connection")
    data_list = cursor.fetchall()
    
    # Convert result to a list of dictionaries
    data_list = [dict(row) for row in data_list]
    
    return jsonify({"result": data_list}), 200



@app.route("/get-list-training-connection/<guid>", methods=["GET"])
def handle_get_list_training_connection(guid):
    db = get_db_connection()
    cursor = db.cursor()
    
    # Fetch the data from the 'training_connection' table
    cursor.execute("SELECT * FROM training_connection WHERE database_guid=?", (guid,))
    data_list = cursor.fetchall()
    
    # Convert result to a list of dictionaries
    data_list = [dict(row) for row in data_list]
    
    return jsonify({"result": data_list}), 200


@app.route("/get-list-thread", methods=["GET"])
def handle_get_list_thread():
    db = get_db_connection()
    cursor = db.cursor()
    
    # Fetch the data from the 'thread' table
    cursor.execute("SELECT * FROM thread")
    data_list = cursor.fetchall()
    
    # Convert result to a list of dictionaries
    data_list = [dict(row) for row in data_list]
    
    # Sort the data by 'created_at' (assuming it's a proper datetime string)
    sorted_data_list = sorted(data_list, key=lambda x: datetime.fromisoformat(x['created_at'].replace("Z", "+00:00")), reverse=True)
    
    return jsonify({"result": sorted_data_list}), 200


@app.route("/get-thread/<guid>", methods=["GET"])
def handle_get_detail_thread(guid):

    if not guid:
        return jsonify({"error": "Missing 'guid' in URL"}), 400

    history = SQLChatMessageHistory(
            session_id=guid, connection_string=f"sqlite:///{db_path}"
        )

    parsed_result = []

    # Iterate through messages and format them
    for message in history.messages:  # Loop through the messages
    # Extract the role from response_metadata['message']['role'] if it exists
        role = message.response_metadata.get("message", {}).get("role", "user")  # Default to 'unknown' if not found
        # Replace 'assistant' with 'system'
        if role == "assistant":
            role = "system"
        
        content = message.content  # Extract the content of the message
    
        # Append the extracted data as a dictionary to the result list
        parsed_result.append({"role": role, "content": content})

    # Convert the parsed result to JSON
    json_result = json.dumps(parsed_result, indent=4, ensure_ascii=False)
          
    try:
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Query the database for the thread details using the given GUID
        cursor.execute(
            """
            SELECT guid, source, type, created_at
            FROM thread_detail
            WHERE guid = ?
            """,
            (guid,)
        )
        thread = cursor.fetchone()

        # Close the connection
        conn.close()

        # If no thread is found, return a 404 error
        if thread is None:
            return jsonify({"error": "Thread not found"}), 404

        # Format the result into a dictionary
        thread_detail = {
            "guid": thread["guid"],
            "source": thread["source"],
            "type": thread["type"],
            "created_at": thread["created_at"]
        }

        # Return the thread details along with the message history
        return jsonify({
            "result": {
                "history": json.loads(json_result),
                "detail": thread_detail
                }
        }), 200

    except Exception as e:
        # Handle any unexpected server errors
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route("/get-list-source", methods=["GET"])
def handle_get_list_core():
    try:
        # Connect to the SQLite database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Fetch data from the custom_prompt table
        cursor.execute("SELECT guid, name FROM custom_prompt")
        list_custom_prompt = cursor.fetchall()

        # Fetch data from the sql_connection table
        cursor.execute("SELECT guid, name FROM sql_connection")
        list_sql_connection = cursor.fetchall()

        # Close the connection
        conn.close()

        # Transforming data into the desired model
        result = []

        # Process the custom prompts
        for item in list_custom_prompt:
            result.append(
                {
                    "guid": item[0],  # GUID is in the first column
                    "type": "custom_prompt",  # Type for custom prompts
                    "name": item[1],  # Name is in the second column
                }
            )

        # Process the SQL connections
        for item in list_sql_connection:
            result.append(
                {
                    "guid": item[0],  # GUID is in the first column
                    "type": "database",  # Type for SQL connections
                    "name": item[1],  # Name is in the second column
                }
            )

        # Return the result as JSON
        return jsonify({"result": result}), 200

    except Exception as e:
        # Handle any unexpected errors
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


# Run Flask app
if __name__ == "__main__":
    # app.run(port=8000, debug=True)
    initSqlite()
    app.run(host="0.0.0.0", port=8000, debug=True)
