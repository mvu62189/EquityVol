import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Parquet Viewer & Converter", layout="wide")

st.title("📊 Parquet to CSV Converter")
st.write("Upload your data to inspect and convert.")

uploaded_files = st.file_uploader("Choose Parquet files", type="parquet", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read Parquet
        df = pd.read_parquet(uploaded_file)
        
        with st.expander(f"📄 {uploaded_file.name}"):
            st.dataframe(df) # Show preview
            
            # Conversion Logic
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=uploaded_file.name.replace('.parquet', '.csv'),
                mime='text/csv',
            )

            # Optional: Quick Stat check
            if st.checkbox(f"Show Statistics for {uploaded_file.name}"):
                st.write(df.describe())