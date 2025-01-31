# config.py
SYSTEM_PREFIX = """Given an input question, create a syntactically correct PostgreSQL query to run.
Always minimum limit the query to at most 5 results unless specified otherwise.
Never perform DML operations (INSERT, UPDATE, DELETE, DROP) on the database.
If the question does not seem related to the database, just return answer like normal AI.
If Return the SQL query only, without additional explanations or comments.
"""

SYSTEM_PREFIX_TOOL = """Execute valid sql query from few_shot_prompt with tool execute_sql.
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
dont show fig.show().
input column.
only final (fig).to_json().
Fix columns of different type.
Fix unable to parse string.
Fix name 'pd' is not defined.
Fix Invalid property specified for object of type plotly.graph_objs.Layout: 'pie'.
Final result (fig).to_json().
Without ```python.
Do not provide change notes.
Return python code only, without additional explanations or comments."""

CLASSIFY_PROMPT = """
Answer directly and concisely in the user's language. Follow the instructions strictly: 
                Classify the input into one of the following categories:
                -  'database': If it involves SQL queries, data structure, tables, or anything technical about databases. 
                  Also classify as 'database' if the prompt mentions "show in table" or similar phrases.
                - 'database_view_chart': If it involves visualizing data, creating graphs, diagrams, or charts.
                - 'general_question': If it is a general question not related to databases or visualization.

                For 'database_view_chart', extract any customization instructions for the chart. 
                Look for the following details if they are mentioned:
                - x-axis: What should be displayed on the x-axis?
                - y-axis: What should be displayed on the y-axis?
                - chart_type: What type of chart (e.g., bar, line, scatter)?
                - title: Any title for the chart?
                - color: Any preferred color or theme for the chart?

                Also, extract the main question from the input.
                Respond in JSON format like this:
                {{
                  "prompt": "<user's original input>",
                  "category": "<one of the categories>",
                  "question": "<the extracted main question for database/chart categories>"
                }}
                For 'general_question', respond as follows:
                {{
                  "prompt": "<user's original input>",
                  "category": "general_question",
                  "question": "This tool only supports questions related to databases and charts.",
                }}
"""