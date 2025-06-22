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
    You have a Pandas DataFrame `df` and Plotly’s `go` already available.  
    Inspect the DataFrame schema and sample rows to choose the most appropriate visualization. Here are some guidelines:

    1. **Select the chart type** based on the columns you find:
       - **Time Series**: If there’s a date/time/datetime column and ≥1 numeric column, create a line chart showing each numeric series over time.
       - **Bar Chart**: If there’s exactly one categorical column and one numeric column, create a simple bar chart.
       - **Grouped Bar Chart**: If there’s one categorical column and multiple numeric columns, create a grouped bar chart (one bar series per numeric column) (`barmode='group'`).
       - **Stacked Bar Chart**: If there’s one categorical column and multiple numeric columns that sum to a meaningful total, create a stacked bar chart (`barmode='stack'`).
       - **Combo Chart**: If there are exactly two numeric columns and one represents percentages (values between 0–1 or name contains “%”), 
         combine a bar chart (for absolute values) with a line‐with‐markers (for percentages) on a secondary y‐axis formatted as percentages.
       - **Scatter Plot**: If there are exactly two numeric columns and no date/time/datetime or categorical column, create a scatter plot.
       - **Histogram**: If there’s one numeric column and no date/time/datetime column, create a histogram.

       Use the above as guidelines. Assess the Pandas DataFrame `df` provided and decide on the most appropriate chart.
    
    2. **Generate dynamic labels**:
       - Derive the chart title by combining the relevant column names (e.g., “Sales by Region Over Time”).
       - Use the actual column names (with underscores replaced by spaces, title‐cased) for the x and y axis labels.
    
    3. **Adjust layout for readability**:
       - Rotate or wrap x‐axis labels if they are long or overlap (e.g., `fig.update_layout(xaxis_tickangle=45,...)`).
       - Position the legend where it doesn’t obscure data.
       - Format any percentage axis with tick labels shown as “0%”, “25%”, etc.
    
    4. **Output**:
       - Return only Python Plotly code that defines `fig`.
       - Do NOT include `fig.show()` or comments.
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
    If there are bullet points, keep it to 3–5 bullet points.
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
    
    Generate 2–3 concise, actionable bullet-point recommendations. Formatting rules:
      • Start each bullet with a verb (e.g., “Investigate…”, “Monitor…”, “Optimize…”).  
      • Each bullet point should end with a period and be separated into its own line.
      • Each numeric value is separated from text by spaces, e.g., “2,303,341.09 followed by Toys at 2,271,034.59”.
      • Do not concatenate words and numbers: e.g., write “2,078,830.10 and 2,040,084.60” not “2,078,830.10and2,040,084.60”.
      • Numeric ranges appear as “X to Y” with a single space on each side of “to”.
    
    Return **only** the bullet list — no extra sentences.
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