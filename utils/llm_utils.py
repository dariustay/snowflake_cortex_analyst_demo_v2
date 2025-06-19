import json
from typing import Dict, Any
import streamlit as st
from snowflake.cortex import Complete, CompleteOptions


def _build_chart_payload(schema_dict: Dict[str, str], sample_rows: Dict[str, Any]) -> str:
    """
    Build the JSON payload instructing the LLM to return Plotly code.
    Payload includes:
      - "schema": { column_name: dtype_as_str }
      - "samples": { column_name: [ first few rows as strings ] }
      - "instructions": Natural-language guidelines on choosing chart type and formatting.
    """
    
    instructions = """
    You have a Pandas DataFrame called df and Plotly’s 'go' is already imported.
    
    1. Inspect the schema (column types) and sample rows to choose the most appropriate chart type, e.g.:
       • If there is a datetime column and at least one numeric column, generate a line chart over time.
       • If there is exactly one categorical (string) column and exactly one numeric column, generate a bar chart.
       • If there are two numeric columns (and no datetime), generate a scatter plot.
       • If there is only one numeric column, generate a histogram.
       • If there is one categorical column with ≤10 distinct values and no numeric column, generate a pie chart.
    
    2. When you generate the Plotly code, ensure:
       • Axis labels and tick labels do not get cut off:
         – For y-axis tick labels (which can be long), use automargin=True, and set larger left margin (e.g., margin=dict(l=80, r=40, t=50, b=60)).
         – For x-axis tick labels, also use automargin=True or rotate long labels (e.g., tickangle=-45).
       • Use a light chart background (e.g., plot_bgcolor='#f9f9f9') so the chart stands out against the app.
       • Keep grid lines visible: xaxis_showgrid=True, yaxis_showgrid=True.
       • Each series or category must use distinct, qualitative colors.
       • Title and axis titles reflect the column names clearly (e.g., “Sales Over Time” or “Category vs Amount”).
       • Font size should be at least 12 for readability.
    
    3. Return only the final Python code snippet (no imports, no comments, no markdown fences) that:
       • Creates the figure: fig = go.Figure().
       • Adds the appropriate go.* trace(s) for the chosen chart type.
       • Do NOT include fig.show().
    """.strip()
    
    payload = {
        "schema": schema_dict,
        "samples": sample_rows,
        "instructions": instructions,
    }
    return json.dumps(payload)


@st.cache_data(show_spinner=False)
def get_chart_code(schema_dict: Dict[str, str], sample_rows: Dict[str, Any], model_name: str, temperature: float, max_tokens: int) -> str:
    """
    Call the LLM to generate Plotly code for visualizing the DataFrame.
    """
    
    payload = _build_chart_payload(schema_dict, sample_rows)
    options = CompleteOptions({
        "temperature": temperature,
        "max_tokens": max_tokens
    })
    try:
        return Complete(model_name, payload, options=options)
    except Exception:
        return ""


def _build_summary_prompt(schema_json: str, sample_json: str, chat_history: str) -> str:
    """
    Construct the prompt for the LLM to produce a structured summary.
    This prompt includes:
      1. A JSON schema of the entire DataFrame (all columns).
      2. A JSON dump of all rows (since the user requested no limit on summary data).
      3. The full chat history for context.
    """
    
    instructions = """
    You are given:
      1. A Pandas DataFrame with schema {schema}.
      2. A set of sample rows: {examples}.
      3. The full chat history between the user and the assistant: {chat_history}.
    
    Use the chat history strictly as context. Provide a clear, structured summary
    of the DataFrame’s key observations. Make sure:
      • There is exactly one space after every comma, period, and colon.
      • Each numeric value is separated from text by spaces, e.g.,
        “2,303,341.09 followed by Toys at 2,271,034.59”.
      • Do not concatenate words and numbers: e.g., write
        “2,078,830.10 and 2,040,084.60” not “2,078,830.10and2,040,084.60”.
      • Numeric ranges appear as “X to Y” with a single space on each side of “to”.
    
    For example:
      - There is a gradual upward trend in revenue from March to May.
      - The lowest daily revenue values range from 9,549.32 to 9,913.01 with minimal variation.
      - No extreme outliers are present; most values stay within a tight band.
    
    Return only the structured summary (following the style shown above). 
    Do NOT mention the user’s original questions or next steps.
    """.strip()

    return instructions.format(
        schema=schema_json,
        examples=sample_json,
        chat_history=chat_history,
    )


@st.cache_data(show_spinner=False)
def get_summary(schema_json: str, sample_json: str, chat_history: str, model_name: str, temperature: float, max_tokens: int) -> str:
    """
    Call the LLM to generate a structured summary of the DataFrame.
    """
    
    prompt = _build_summary_prompt(schema_json, sample_json, chat_history)
    options = CompleteOptions({
        "temperature": temperature,
        "max_tokens": max_tokens
    })
    try:
        return Complete(model_name, prompt, options=options)
    except Exception:
        return "Summary is unavailable at the moment."
        

def _build_insights_prompt(schema_json: str, sample_json: str, chat_history: str) -> str:
    """
    Construct the LLM prompt for generating actionable insights based on a DataFrame.
    """
    
    instructions = instructions = """
    You are given:
      1. A dataset schema: {schema}
      2. Sample rows: {examples}
      3. The chat history: {chat_history}
    
    Generate **3–5** concise, **actionable** bullet-point recommendations. **Formatting rules**:
      • Start each bullet with a verb (e.g., “Investigate…”, “Monitor…”, “Optimize…”).  
      • Always use exactly one space before and after parentheses, e.g., “January (55,919) to February (42,299)”.  
      • Represent numeric ranges with an en-dash **without** spaces, e.g., “46,000–50,000”.  
      • Include the “$” currency symbol immediately before numbers (no extra spaces).  
    Return **only** the bullet list—no extra sentences or numbering.
    """.strip()
    return instructions.format(
        schema=schema_json,
        examples=sample_json,
        chat_history=chat_history,
    )


@st.cache_data(show_spinner=False)
def get_insights(schema_json: str, sample_json: str, chat_history: str, model_name: str, temperature: float, max_tokens: int) -> str:
    """
    Call the LLM to generate actionable insights for the user based on a DataFrame.
    """
    
    prompt = _build_insights_prompt(schema_json, sample_json, chat_history)
    options = CompleteOptions({"temperature": temperature, "max_tokens": max_tokens})
    try:
        return Complete(model_name, prompt, options=options)
    except Exception:
        return "Insights are unavailable at the moment."