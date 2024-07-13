import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import random
import logging
from io import StringIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log_stream = StringIO()
handler = logging.StreamHandler(log_stream)
logger.addHandler(handler)

# Loading environment variables from .env file
load_dotenv()

# Function to chat with CSV data
def chat_with_csv(df, query):
    try:
        logger.info("Initializing AI model...")
        groq_api_key = os.environ['GROQ_API_KEY']
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=0.1)
        pandas_ai = SmartDataframe(df, config={"llm": llm})
        logger.info("Processing query with AI model...")
        result = pandas_ai.chat(query)
        logger.info("Query processed successfully.")
        return result
    except Exception as e:
        error_message = f"Error: {str(e)}"
        logger.error(error_message)
        return error_message

# Function to generate dynamic sample queries
def generate_sample_queries(columns):
    sample_queries = [
        f"What is the total count of {random.choice(columns)}?",
        f"Show the average value of {random.choice(columns)} over time.",
        f"Which {random.choice(columns)} has the highest value?",
        f"Compare the {random.choice(columns)} between different categories."
    ]
    return sample_queries

# Set layout configuration for the Streamlit page
st.set_page_config(page_title="CSV Chat Hackathon 🧑‍💻", layout='wide')

# Set title for the Streamlit application
st.title("CleverQuery🧑‍💻")
st.write("An AI-powered tool to interact with CSV data using natural language queries.")

# Upload multiple CSV files
st.sidebar.header("Upload Files")
input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# Check if CSV files are uploaded
if input_csvs:
    try:
        with st.spinner("Loading CSV files..."):
            logger.info("CSV files uploaded.")
            selected_file = st.sidebar.selectbox("Select a CSV file", [file.name for file in input_csvs])
            selected_index = [file.name for file in input_csvs].index(selected_file)

            # Load and display the selected CSV file
            data = pd.read_csv(input_csvs[selected_index])
            st.sidebar.success("CSV uploaded successfully")

            # Display data summary
            st.subheader("Data Summary")
            st.write(data.describe())

            # Display the data frame
            st.subheader("Data Preview")
            st.dataframe(data.head(), use_container_width=True)

            # Sidebar for basic statistics
            if st.sidebar.checkbox("Show Basic Statistics"):
                st.subheader("Basic Statistics")
                st.write(data.describe())

            # Initialize session state for sample queries
            if 'sample_queries' not in st.session_state:
                st.session_state.sample_queries = generate_sample_queries(data.columns)

            # Generate and display dynamic sample queries
            sample_queries = st.session_state.sample_queries
            st.sidebar.subheader("Sample Queries")
            selected_sample_query = st.sidebar.radio("Select a sample query to execute:", sample_queries)

            # Enter the query for analysis
            st.subheader("Chat with CSV")
            input_text = st.text_area("Enter your query or select a sample query from the sidebar:")

            # Perform analysis
            if st.button("Chat with CSV") or st.sidebar.button("Execute Selected Sample Query"):
                query = input_text if input_text else selected_sample_query
                st.info("Your Query: " + query)
                with st.spinner("Processing query..."):
                    result = chat_with_csv(data, query)
                if "Error:" in result:
                    st.error(result)
                else:
                    st.success(result)
                # Update sample queries after executing the selected query
                st.session_state.sample_queries = generate_sample_queries(data.columns)

    except Exception as e:
        error_message = f"Error loading CSV file: {str(e)}"
        st.sidebar.error(error_message)
        logger.error(error_message)
else:
    st.sidebar.warning("Please upload CSV files to proceed.")

# Add more information and instructions for users
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload one or more CSV files.
2. Select the CSV file from the dropdown menu.
3. View basic statistics and data preview.
4. Enter your query or select a sample query from the sidebar.
5. Interact with the CSV data using AI.
""")

# Add download option for logs
st.sidebar.header("Download Logs")
if st.sidebar.button("Download Logs"):
    log_contents = log_stream.getvalue()
    st.sidebar.download_button(label="Download Logs", data=log_contents, file_name='logs.txt', mime='text/plain')


# Add footer
st.sidebar.markdown("""
---
Developed by Aditya Jethani
""")
