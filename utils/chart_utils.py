import textwrap
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def sanitize_plotly_code(raw_code: str) -> str:
    """
    Strip out any `fig.show()` calls (or similar) so that executing the code
    won't attempt to open an interactive window.
    """
    
    return raw_code.replace("fig.show()", "").strip()


def execute_plotly_code(df: pd.DataFrame, raw_plotly_code: str) -> go.Figure:
    """
    Execute the sanitized Plotly code in a namespace that provides:
      - df (the DataFrame)
      - px (plotly.express)
      - go (plotly.graph_objects)

    Expects the code to define a variable named `fig`. Returns that Figure.
    """
    
    code = sanitize_plotly_code(raw_plotly_code)
    code = textwrap.dedent(code)
    local_vars = {"df": df, "px": px, "go": go}
    exec(code, {}, local_vars)
    fig = local_vars.get("fig")
    if fig is None:
        raise ValueError("No `fig` object was created by the Plotly code.")
    return fig
