import json
import streamlit as st
import _snowflake
import pandas as pd

from typing import Any, Dict, List

from utils.snowflake_utils import session, get_cached_df
from utils.llm_utils import get_chart_code, get_summary
from utils.chart_utils import execute_plotly_code


# === Configuration Constants ===

DATABASE = "CORTEX_ANALYST_DEMO_V2"
SCHEMA   = "REVENUE_TIMESERIES"
STAGE    = "RAW_DATA"
FILE     = "revenue_timeseries.yaml"

MAX_SAMPLE_ROWS = 10

MODEL_NAME_SUMMARY = "claude-3-5-sonnet"
MODEL_NAME_CHART   = "claude-3-5-sonnet"

SUMMARY_TEMPERATURE = 0.1
SUMMARY_MAX_TOKENS  = 500

CHART_TEMPERATURE = 0.1
CHART_MAX_TOKENS  = 750


# === Helper Functions ===

def send_message(prompt: str) -> Dict[str, Any]:
    """
    Send a user prompt with the chat history to the Snowflake Cortex Analyst API and return the parsed JSON response.
    """
    
    messages_payload = [
        {
            "role": "analyst" if msg["role"] == "assistant" else msg["role"],
            "content": msg["content"]
        }
        for msg in st.session_state.messages
    ]

    request_body = {
        "messages": messages_payload,
        "semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}",
    }

    resp = _snowflake.send_snow_api_request(
        "POST",
        "/api/v2/cortex/analyst/message",
        {},
        {},
        request_body,
        None,
        50000,
    )

    if resp["status"] < 400:
        return json.loads(resp["content"])
    else:
        raise Exception(f"Failed request with status {resp['status']}: {resp}")


def join_chat_history() -> str:
    """
    Convert all chat messages in st.session_state.messages into one newline-separated string of "ROLE: text" lines.
    """
    
    lines: List[str] = []
    for msg in st.session_state.messages:
        role = msg["role"].upper()
        for chunk in msg["content"]:
            if chunk["type"] == "text":
                text = chunk["text"].strip()
                lines.append(f"{role}: {text}")
    return "\n".join(lines)


def build_chart_sample(df: pd.DataFrame, max_rows: int = MAX_SAMPLE_ROWS) -> Dict[str, List[str]]:
    """
    Build a “lightweight” sample of the DataFrame for the chart prompt.
    We send ALL columns but only the first `max_rows` rows (stringified).
    """
    
    sample_df = df.head(max_rows).astype(str)
    return sample_df.to_dict(orient="list")


def display_data_and_chart(df: pd.DataFrame) -> None:
    """
    Given a Pandas DataFrame, render:
      • a “Data” tab with st.dataframe(df)
      • a “Chart” tab that:
           – Builds a limited‐row sample of all columns
           – Calls get_chart_code(...) with chart‐specific LLM parameters
           – Tries exec‐ing that code 3 times via execute_plotly_code()
           – If successful, shows the Plotly figure + an expander with the raw code
           – Otherwise, shows an error
      • a “Summary” tab that:
           – Sends ALL columns & ALL rows to get_summary(...) with summary‐specific LLM parameters
           – Renders the LLM’s structured summary

    If df has 0 or 1 rows, we skip Chart+Summary and simply show the table.
    """
    
    # If there are no rows or only a single row, just display the DataFrame
    if df is None or df.shape[0] <= 1:
        st.dataframe(df if df is not None else pd.DataFrame())
        return

    # Build schema_dict for the LLM: { column_name: dtype_as_string }
    schema_dict = {col: str(dtype) for col, dtype in df.dtypes.astype(str).items()}

    # Build the lightweight sample for chart generation
    sample_rows_chart = build_chart_sample(df)

    # Create two tabs: “Data” and “Chart”
    tabs = st.tabs(["Data", "Chart"])

    # Tab 1: Show the raw DataFrame
    with tabs[0]:
        st.dataframe(df)

    # Tab 2: Show the chart or an error if code fails
    with tabs[1]:
        with st.spinner("Rendering chart…"):
            chart_success = False
            fig = None
            
            # Try up to 3 times to generate abd execute the Plotly code
            for attempt in range(3):
                try:
                    # Generate Plotly code via LLM (pass chart‐param constants)
                    plotly_code = get_chart_code(
                        schema_dict=schema_dict,
                        sample_rows=sample_rows_chart,
                        model_name=MODEL_NAME_CHART,
                        temperature=CHART_TEMPERATURE,
                        max_tokens=CHART_MAX_TOKENS,
                    )
    
                    fig = execute_plotly_code(df, plotly_code)
                    chart_success = True
                    break
                except Exception:
                    if attempt < 2:
                        continue
    
            if not chart_success or fig is None:
                st.error("Chart is not available.")
            else:
                # Show raw Python code in a collapsible expander
                with st.expander("View Chart Code", expanded=False):
                    st.code(plotly_code, language="python")
                    
                # Display the Plotly figure
                st.plotly_chart(fig, use_container_width=True)

    # Summary tab (send ALL rows & ALL columns to the LLM)
    summary_tabs = st.tabs(["Summary"])
    with summary_tabs[0]:
        
        # Convert entire DataFrame to JSON-friendly string‐dict
        sample_rows_summary = df.astype(str).to_dict(orient="list")

        # Combine full chat history as a single string
        chat_history_str = join_chat_history()

        # Ask the LLM for a structured summary (pass summary‐specific constants)
        with st.spinner("Generating summary…"):
            summary = get_summary(
                schema_json=json.dumps(schema_dict),
                sample_json=json.dumps(sample_rows_summary),
                chat_history=chat_history_str,
                model_name=MODEL_NAME_SUMMARY,
                temperature=SUMMARY_TEMPERATURE,
                max_tokens=SUMMARY_MAX_TOKENS,
            )

        st.markdown(summary)


def render_sql_item(item: Dict[str, Any]) -> None:
    """
    Handle a single LLM-returned SQL block:
      1) Show raw SQL in a collapsed expander
      2) Execute SQL (cached via get_cached_df) → Pandas DataFrame
      3) Call display_data_and_chart() on the resulting DataFrame
    """
    
    sql = item["statement"]

    # Show raw SQL in a collapsed expander
    with st.expander("SQL Query", expanded=False):
        st.code(sql, language="sql")

    # Execute SQL (cached) to get a DataFrame
    try:
        df = get_cached_df(sql)
    except Exception as e:
        st.error(f"SQL execution failed: {e}")
        return

    # Display Data + Chart + Summary
    display_data_and_chart(df)


def render_message(content: List[Dict[str, Any]], message_index: int) -> None:
    """
    Render a single assistant message (or replayed user message) in the Streamlit chat UI.

    Each chunk in `content` can be:
      - {"type": "text", "text": "..."}        → render st.markdown(...)
      - {"type": "suggestions", "suggestions": [...] } → render buttons
      - {"type": "sql", "statement": "<SQL>"} → call render_sql_item()
    """
    
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])
        elif item["type"] == "suggestions":
            with st.expander("Suggestions", expanded=True):
                for idx, suggestion in enumerate(item["suggestions"]):
                    if st.button(suggestion, key=f"{message_index}_{idx}"):
                        st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            render_sql_item(item)


def process_message(prompt: str) -> None:
    """
    Handle a new user chat message:
      1) Append the user’s message to st.session_state.messages
      2) Echo it in the UI via st.chat_message("user")
      3) Call send_message() to get assistant’s response
      4) Render each piece of assistant’s response via render_message()
      5) Append assistant’s response to st.session_state.messages
    """
    
    # Append user message to history
    st.session_state.messages.append(
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    )

    # Echo user message in chat UI
    st.chat_message("user").markdown(prompt)

    # Call the Cortex Analyst API
    with st.spinner("Generating response…"):
        try:
            response = send_message(prompt)
        except Exception as e:
            st.error(f"LLM request failed: {e}")
            return

    # Render assistant’s content
    assistant_content = response["message"]["content"]
    with st.chat_message("assistant"):
        render_message(assistant_content, len(st.session_state.messages))

    # Append assistant’s response to history
    st.session_state.messages.append({"role": "assistant", "content": assistant_content})


# === Streamlit App  ===

def main():

    # Sidebar: Show the semantic model file
    st.sidebar.title("Configuration")
    st.sidebar.markdown(f"**Semantic model**: `{FILE}`")
    st.sidebar.markdown("---")

    # Main title
    st.title("Revenue Timeseries Explorer")

    # A short introduction below the title
    st.markdown(
        """
        Welcome to the Revenue Timeseries Explorer!

        This app lets you:
        - 🔍 **Explore** daily revenue trends by region and product  
        - 📊 **Visualize** data with dynamic charts  
        - 💬 **Ask** natural‐language questions and get instant insights  
        """
    )
    st.write("")

    # Initialize chat history if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.active_suggestion = None

    # Replay any prior messages
    for msg_index, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            render_message(msg["content"], msg_index)

    # Show the chat input box at the bottom
    if user_input := st.chat_input("What is your question?"):
        process_message(user_input)

    # If a suggestion button was clicked, process it
    if st.session_state.active_suggestion:
        process_message(st.session_state.active_suggestion)
        st.session_state.active_suggestion = None


if __name__ == "__main__":
    main()