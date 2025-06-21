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
    
    1. Inspect schema and data to pick the most appropriate chart. Here are some guidelines:
        • **Time series**: If there’s a datetime column and ≥1 numeric, use line(s) over time.
        • **Combo**: If there are exactly two numeric columns and one is percent (values in [0,1] or name contains '%'), use:
           – go.Bar for the absolute values on the left y-axis.
           – go.Scatter (mode='lines+markers') for the percentage on a secondary y-axis (`overlaying='y'`, `side='right'`, `tickformat='.0%'`).
        • **Grouped bar**: If one categorical + >1 numeric, plot grouped bars.
        • **Scatter**: If exactly two numerics and no datetime, scatter plot.
        • **Histogram**: If one numeric and no datetime, histogram.
        
    2. When generating code:
        • Build dynamic titles and axis names from column headers.
        • Rotate or wrap x-axis labels based on type and length.
        • Use `plot_bgcolor='#f9f9f9'`, `automargin=True`, and sensible margins.
        • Ensure legend and gridlines are visible.
        • Return only Python code that defines `fig`.
    
    3. Do NOT include `fig.show()` or comments.
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