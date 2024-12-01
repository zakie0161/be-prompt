from flask import Flask, request, jsonify, session
from flask_session import Session
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List
from sqlalchemy import create_engine
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
import pandas as pd
import re
from urllib.parse import quote
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.tools import PythonAstREPLTool
from langchain.output_parsers.pandas_dataframe import PandasDataFrameOutputParser
import os
import json
import random
import string
from datetime import datetime

# Flask app
app = Flask(__name__)
app.secret_key = 'my_secret_key'

# Configure Flask-Session to use server-side sessions
app.config['SESSION_TYPE'] = 'filesystem'  # Could be 'redis', 'mongodb', etc.
app.config['SESSION_PERMANENT'] = False

# Initialize the session extension
Session(app)

encoded_password = quote("P@ssw0rd2022!!")

# URL koneksi PostgreSQL Anda
database_url = f"postgresql://postgres:P@ssw0rd2022!!@34.128.119.53:5432/alaya-copy"
database_url_encoded = f"postgresql://postgres:{encoded_password}@34.128.119.53:5432/alaya-copy"

engine = create_engine(database_url_encoded)

base_url= "https://2c2d-34-142-194-202.ngrok-free.app"

# LangChain setup
llm = ChatOllama(
    model="llama3.1",
    # model="llama3.2:3b-instruct-fp16",
    temperature=0,
    base_url=base_url,
)

system_prefix = """Given an input question, create a syntactically correct PostgreSQL query to run.
Always minimum limit the query to at most 5 results unless specified otherwise.
Never perform DML operations (INSERT, UPDATE, DELETE, DROP) on the database.
If the question does not seem related to the database, just return "I don't know" as the answer.
Return the SQL query only, without additional explanations or comments.
"""

examples = [
    {"input": "List all artists.", "query": "SELECT * FROM Artist;", "description": ""},
    {"input": "List all cms.", "query": "SELECT * FROM cms;", "description": ""},
    {"input": "List all transactions.", "query": "SELECT * FROM transaction_schema.transactions limit 10;", "description": ""},
    {"input": "List all sidebar menu.", "query": "SELECT * FROM sidebar_menu;", "description": ""},
    {"input": "List all last month.", "query": "SELECT * FROM last_month;", "description": ""},
    {"input": "Top Product Therapist.", "query": "SELECT report_schema.report_top_product_therapist('5cec917d-2aab-48bb-923b-8642a8a3c1cc', false, null , null);", "description": ""}
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OllamaEmbeddings(model='nomic-embed-text', base_url=base_url),
    FAISS,
    k=5,
    input_keys=["input"],
)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "{query}"
    ),
    input_variables=["input"],
    prefix=system_prefix,
    suffix="",
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
    ]
)

layout_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
         You are an AI assistant tasked with processing JSON data and generating output based on my request. I will provide data in JSON format and ask you to perform one of two actions:
   
1. Show in Chart (Bar Chart, Line Chart, or Scatter Plot): Transform the JSON data into a Plotly-compatible JSON format with 'data' and 'layout' keys. 'data' should contain chart trace information, and 'layout' should contain chart configuration, like title and axis labels.

Here is the provided JSON data:

{json_input}

Example Output (Bar Chart):
{{
    "data": [
        {{
            "type": "bar",
            "x": ["Alice", "Bob", "Charlie"],
            "y": [25, 30, 35],
            "name": "Age"
        }}
    ],
    "layout": {{
        "title": "Age by Name",
        "xaxis": {{"title": "Name"}},
        "yaxis": {{"title": "Age"}}
    }}
}}

Output without any explanations or additional commentary.
Result JSON format only and without any explanations or additional commentary.
Please process the following JSON data and generate output based on my request.
         
        """),
        ("human", "{input}"),
    ]
)

custom_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
         Jawab langsung ke intinya, jawab dengan bahasa user dan jawab sesuai instruksi: 
         {custom_prompt}"""),
        ("human", "{input}"),
    ]
)

chain = full_prompt | llm
chain_layout = layout_prompt | llm
chain_custom_prompt = custom_prompt | llm

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

def generateQuery(input):
    result = chain.invoke({"input": input})
    sql = clean_generation_result(result.content)
    return sql

def generateJson(input):

    sql_query = session.get('current_query', '')
    print(sql_query)
    if is_sql_query(sql_query):
      data = pd.read_sql(sql_query, engine)  # Execute the SQL query against the database
      df = pd.DataFrame(data)

    else:
      df = pd.DataFrame([sql_query])

    # result = chain.invoke({"question": input})
    result = chain_layout.invoke({"input": input, "json_input": df.head().to_json(orient="split")})
    sql = result.content
    return sql

def generateFromCustomPrompt(input, custom_prompt):
    result = chain_custom_prompt.invoke({"input": input, "custom_prompt":custom_prompt})
    return result.content

def is_sql_query(query):
    sql_keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY', 'LIMIT', 'VALUES', 'INTO']
    query = query.upper()
    return any(re.search(r'\b' + re.escape(keyword) + r'\b', query) for keyword in sql_keywords)

# Function to generate a random string
def generate_random_string(length):
    # Choose random characters from letters and digits
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


# Flask endpoint for generateQuery
@app.route('/generate-query', methods=['POST'])
def handle_generate_query():
    data = request.json
    input_query = data.get("input")
    
    if not input_query:
        return jsonify({"error": "Missing input query"}), 400
    
    sql = generateQuery(input_query)

    session['current_query'] = sql
   
    return jsonify({"result": sql}), 200


@app.route('/generate-json', methods=['POST'])
def handle_generate_json():
    # Get the JSON payload from the request
    data = request.json
    input_query = data.get("input")  # Extract the 'input' field
    
    # Check if both 'input' and 'sql_input' are provided
    if not input_query:
        return jsonify({"error": "Missing input query or SQL query"}), 400
    
    # Assuming you have a function 'generateJson' to process the queries
    try:
        result = generateJson(input_query)  # This should return the desired JSON
    except Exception as e:
        # Handle any errors that might occur in the 'generateJson' function
        return jsonify({"error": f"Error generating JSON: {str(e)}"}), 500
    
    parsed_result = json.loads(result)
    # Return the result from generateJson in the response
    return jsonify({"result": parsed_result}), 200

@app.route('/generate-from-custom-prompt', methods=['POST'])
def handle_generate_from_custom_prompt():
    # Get the JSON payload from the request
    data = request.json
    input_query = data.get("input")  # Extract the 'input' field
    custom_prompt = data.get("custom_prompt")  # Extract the 'input' field
    
    # Check if both 'input' and 'sql_input' are provided
    if not input_query or not custom_prompt:
        return jsonify({"error": "Missing input or custom prompt"}), 400
    
    # Assuming you have a function 'generateJson' to process the queries
    result = generateFromCustomPrompt(input_query, custom_prompt)
    # Return the result from generateJson in the response
    return jsonify({"result": result}), 200

@app.route('/create-custom-prompt', methods=['POST'])
def handle_create_custom_prompt():
    # Get the JSON payload from the request
    data = request.json
    name = data.get("name")  # Extract the 'input' field
    custom_prompt = data.get("custom_prompt")  # Extract the 'input' field
    
    # Check if both 'input' and 'sql_input' are provided
    if not name or not custom_prompt:
        return jsonify({"error": "Missing input or custom prompt"}), 400

    prompts = session.get('custom_prompt', [])

    now = datetime.now()

    # Convert to string with a literal 'Z' at the end
    date_string = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    data["guid"] = generate_random_string(10)
    data["created_at"] = date_string

    # Add the new prompt to the list
    prompts.append(data)

    session['custom_prompt'] = prompts

    # Return the result from generateJson in the response
    return jsonify({"result": 'Success'}), 200

@app.route('/get-list-custom-prompt', methods=['GET'])
def handle_get_list_custom_prompt():
    # Remove the JSON list from the session (GET method)
    data_list = session.get('custom_prompt', [])
    return jsonify({"result": data_list}), 200

@app.route('/set-ai-host', methods=['POST'])
def handle_set_ai_host():
    # Get the JSON payload from the request
    data = request.json
    host = data.get("host")  # Extract the 'input' field
    
    # Check if both 'input' and 'sql_input' are provided
    if not host:
        return jsonify({"error": "Missing host"}), 400

    session['host'] = host

    # Return the result from generateJson in the response
    return jsonify({"result": 'Success'}), 200

@app.route('/get-ai-host', methods=['GET'])
def get_ai_host():
    # Remove the JSON list from the session (GET method)
    host = session.get('host', '')
    return jsonify({"result": host}), 200

@app.route('/create-sql-connection', methods=['POST'])
def handle_create_sql_connection():
    # Get the JSON payload from the request
    data = request.json
    # Check if both 'input' and 'sql_input' are provided
    required_fields = ['name', 'database_name', 'user', 'password', 'host', 'port']

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing field!"}), 400

    prompts = session.get('sql_connection', [])

    now = datetime.now()

    # Convert to string with a literal 'Z' at the end
    date_string = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    data["guid"] = generate_random_string(10)
    data["created_at"] = date_string

    # Add the new prompt to the list
    prompts.append(data)

    session['sql_connection'] = prompts

    # Return the result from generateJson in the response
    return jsonify({"result": 'Success'}), 200

@app.route('/get-list-sql-connection', methods=['GET'])
def handle_get_list_sql_connection():
    # Remove the JSON list from the session (GET method)
    data_list = session.get('sql_connection', [])
    return jsonify({"result": data_list}), 200

# Run Flask app
if __name__ == '__main__':
    app.run(port=8000, debug=True)
