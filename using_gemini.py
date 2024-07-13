import streamlit as st
import pandas as pd
import sqlite3
import google.generativeai as genai
import os
import re
from datetime import datetime

from dotenv import load_dotenv
import plotly.express as px

# Configuration
# Load environment variables from .env file
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CSV_FILE = 'Default CSVs\merged.csv'

# Data processing functions
def load_data(file_path):
    return pd.read_csv(file_path)

def convert_date(date_str):
    try:
        return datetime.strptime(date_str, '%d/%b/%Y').strftime('%Y-%m-%d')
    except ValueError:
        return date_str

def prepare_data(df):
    date_columns = ['DateofEncashment', 'JournalDate', 'DateofPurchase', 'DateofExpiry']
    for col in date_columns:
        df[col] = df[col].apply(convert_date)
    return df

def create_database(df):
    conn = sqlite3.connect(':memory:')
    df.to_sql('electoral_bonds', conn, index=False)
    return conn

# AI query generation
def configure_ai():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel('gemini-pro') #gemini-1.5-flash

def clean_sql_query(query):
    return re.sub(r'```sql|```', '', query).strip()

def generate_sql_query(model, user_input, columns):
    prompt = f"""
    Generate a SQLite-compatible SQL query for the following request: {user_input}
    
    Table name: electoral_bonds
    Columns: {', '.join(columns)}
    
    Important notes:
    1. Do not use any markdown formatting or code blocks.
    2. The 'Denominations_left' and 'Denominations_right' columns contain string values like "10,00,000" or "1,00,00,000".
       Use this SQLite function to convert them to numbers: CAST(REPLACE(REPLACE(column, ',', ''), '"', '') AS INTEGER)
    3. Date columns (DateofEncashment, JournalDate, DateofPurchase, DateofExpiry) are stored in 'YYYY-MM-DD' format.
       Use SQLite date functions for date operations, e.g., DATE(DateofPurchase) for comparisons.
    4. Only return the SQL query, nothing else.
    5. Use CTEs (WITH clause) for complex queries instead of derived tables.
    6. Ensure all column names and table names are correctly referenced.
    """
    response = model.generate_content(prompt)
    return clean_sql_query(response.text)

# Query execution
def execute_query(conn, query):
    try:
        result = pd.read_sql_query(query, conn)
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}\nQuery: {query}"

# Visualization functions
def plot_data(df, x_col, y_col, title):
    fig = px.bar(df, x=x_col, y=y_col, title=title)
    return fig

# Streamlit UI
def main():
    st.set_page_config(page_title="Electoral Bonds Query System", layout="wide")

    st.title("Electoral Bonds Query System")

    # Sidebar
    st.sidebar.header("Options")
    show_raw_data = st.sidebar.checkbox("Show Raw Data")
    show_query = st.sidebar.checkbox("Show Generated SQL Query")

    # Load and prepare data
    df = load_data(CSV_FILE)
    df = prepare_data(df)
    conn = create_database(df)
    model = configure_ai()

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Query Input")
        user_input = st.text_area("Enter your query:", height=100)
        if st.button("Execute Query"):
            if user_input:
                sql_query = generate_sql_query(model, user_input, df.columns)
                if show_query:
                    st.code(sql_query, language="sql")
                result = execute_query(conn, sql_query)
                
                if isinstance(result, pd.DataFrame):
                    st.subheader("Query Results")
                    st.dataframe(result)

                    # Visualization
                    if len(result.columns) >= 2:
                        st.subheader("Visualization")
                        x_col = st.selectbox("Select X-axis", result.columns)
                        y_col = st.selectbox("Select Y-axis", result.columns)
                        title = st.text_input("Chart Title", "Query Results Visualization")
                        fig = plot_data(result, x_col, y_col, title)
                        st.plotly_chart(fig)
                else:
                    st.error(result)
            else:
                st.warning("Please enter a query.")

    with col2:
        st.subheader("Sample Queries")
        sample_queries = [
            "What is the total amount of bonds purchased by the top 5 political parties?",
            "Show the trend of bond purchases over time",
            "Which companies have purchased the highest value of bonds?",
            "Compare the bond values for different political parties"
        ]
        for query in sample_queries:
            if st.button(query):
                st.session_state.user_input = query
                st.experimental_rerun()

    # Show raw data if checkbox is selected
    if show_raw_data:
        st.subheader("Raw Data")
        st.dataframe(df)

if __name__ == "__main__":
    main()