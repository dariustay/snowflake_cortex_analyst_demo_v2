# snowflake_utils.py
import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session


@st.cache_resource
def get_snowflake_session():
    """
    Create (and cache) a Snowpark session. 
    Streamlit reuses this object across reruns to avoid reconnecting each time.
    """
    
    return get_active_session()


# Initialize session
session = get_snowflake_session()


@st.cache_data(show_spinner=False)
def get_cached_df(sql: str) -> pd.DataFrame:
    """
    Execute a SQL query against Snowflake and return a Pandas DataFrame.
    Results are cached (keyed by the SQL string), so identical queries don’t re‐fetch.
    """
    
    df = session.sql(sql).to_pandas()
    return df