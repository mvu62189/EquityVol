import streamlit as st

st.set_page_config(page_title="EquityVolSurface", layout="wide")

st.title("Equity Volatility Surface Manager")
st.markdown("""
**System Status:**
* **ETL:** Ready (yfinance)
* **Analytics:** Ready (Black-Scholes + SVI)
* **Visualization:** Active
""")

st.info("Select a page from the sidebar to analyze snapshots.")