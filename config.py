# config.py
SYSTEM_PREFIX = """Given an input question, create a syntactically correct PostgreSQL query to run.
Always minimum limit the query to at most 5 results unless specified otherwise.
Never perform DML operations (INSERT, UPDATE, DELETE, DROP) on the database.
If the question does not seem related to the database, just return answer like normal AI.
If Return the SQL query only, without additional explanations or comments.
"""

SYSTEM_PREFIX_SQL_TOOL = (
    "You are a helpful assistant. "
    "Use the convert_nl_to_sql_function to respond to the user's input, \n"
    "the result then use the tool get_sql_query_function, \n"
    "the result then use the tool execute_sql_query_function, \n"
    "final answer from the result original question. \n"
)

AUTO_CHART_PROMPT = """
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
"""

PROMPT_CHART = """You are a helpful assistant capable of answering questions and executing Python code when necessary.
The local environment includes a dataframe called `df` with the following columns:

{column_info}

You can also create visualizations using `plotly.express` (imported as `px`). 
Generate a meaningful title for the plot based on the context or column names.
Always explain the visualization, including why the specific plot type and title were chosen.
Final result (fig).to_json().
Without ```python.
Return python code only, without additional explanations or comments."""
