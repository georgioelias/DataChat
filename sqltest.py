import streamlit as st
import pandas as pd
import sqlite3
import json
from openai import OpenAI
import os
import chardet
import io
import re
from collections import Counter

# Constants
DB_NAME = 'data.db'
FIXED_TABLE_NAME = "uploaded_data"

# OpenAI API setup
API_KEYS = [
    st.secrets["OPENAI_API_KEY"],
    st.secrets["OPENAI_API_KEY_2"],
    # Add more backup keys as needed
]
MODELS = ["gpt-4o", "gpt-4o"]  # Add more models as needed

# Global variables for prompts
if 'sql_generation_prompt' not in st.session_state:
    st.session_state.sql_generation_prompt = '''
    User's explanation of the CSV:
    {csv_explanation}

    A user will now chat with you. Your task is to transform the user's request into an SQL query that retrieves exactly what they are asking for.

    Rules:
    1. Return only two JSON variables: "Explanation" and "SQL".
    2. No matter how complex the user question is, return only one SQL query.
    3. Always return the SQL query in a one-line format.
    4. Consider the chat history when generating the SQL query.
    5. The query can return multiple rows if appropriate for the user's question.
    6. You shall not use functions like MONTH(),HOUR(),HOUR,YEAR(),DATEIFF(),.....
    7. Use only queries proper for sql LITE
    8.⁠ ⁠YOU CAN ONLY RETURN ONE SQL STATEMENT AT A TIME, COMBINE YOUR ANSWER IN ONLY ONE STATEMENT, NEVER 2 or MORE, Find workarounds.
    9.Ignore Null values in interpretations and calculations only consider them where they are relevent.


    Example output:
    {{
    "Explanation": "The user is asking about the top 5 users by age. To retrieve this, we need to select the name and age columns, order by age descending, and limit to 5 results.",
    "SQL": "SELECT name, age FROM {table_name} ORDER BY age DESC LIMIT 5"
    }}

    Your prompt ends here. Everything after this is the chat with the user. Remember to always return the accurate SQL query.
    '''

if 'response_generation_prompt' not in st.session_state:
    st.session_state.response_generation_prompt = '''
    User's explanation of the CSV:
    {csv_explanation}

    Now you will receive a JSON containing the SQL output that answers the user's inquiry. The output may contain multiple rows of data. Your task is to use the SQL's output to answer the user's inquiry in plain English. Consider the chat history when generating your response. If there are multiple results, summarize them appropriately.
    '''

############################################### HELPER FUNCTIONS ########################################################
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    return chardet.detect(raw_data)['encoding']

def get_data_type(values):
    if all(isinstance(val, (int, float)) for val in values if pd.notna(val)):
        return "Number"
    elif all(isinstance(val, str) for val in values if pd.notna(val)):
        return "Text"
    elif all(pd.to_datetime(val, errors='coerce') is not pd.NaT for val in values if pd.notna(val)):
        return "Date"
    else:
        return "Mixed"

def analyze_csv(file_path, max_examples=3):
    encoding = detect_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    
    prompt = "This CSV file contains the following columns:\n\n"
    
    for col in df.columns:
        values = df[col].dropna().tolist()
        data_type = get_data_type(values)
        
        unique_count = df[col].nunique()
        total_count = len(df)
        is_unique = unique_count == total_count
        
        examples = df[col].dropna().sample(min(max_examples, len(values))).tolist()
        
        prompt += f"Column: {col}\n"
        prompt += f"Data Type: {data_type}\n"
        prompt += f"Examples: {', '.join(map(str, examples))}\n"
        
        if is_unique:
            prompt += "Note: This column contains unique values for each row.\n"
        
        null_count = df[col].isnull().sum()
        if null_count > 0:
            prompt += f"Note: This column contains {null_count} NULL values.\n"
        
        if data_type == "Text":
            value_counts = Counter(values)
            most_common = value_counts.most_common(3)
            if len(most_common) < len(value_counts):
                prompt += f"Most common values: {', '.join(f'{val} ({count})' for val, count in most_common)}\n"
        
        prompt += "\n"
    
    return prompt

def reset_chat():
    st.session_state.messages = []

def display_sql_query(query):
    with st.expander("View SQL Query", expanded=False):
        st.code(query, language="sql")

def display_json_data(json_data):
    with st.expander("View JSON Data", expanded=False):
        if isinstance(json_data, list):
            for item in json_data:
                st.json(item)
        else:
            st.json(json_data)

def df_to_sqlite(df, table_name, db_name=DB_NAME):
    try:
        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"An error occurred while creating the table: {e}")
        return False

# New function to update prompts
def update_prompt(prompt_type):
    if prompt_type == "SQL Generation":
        st.session_state.sql_generation_prompt = st.session_state.sql_generation_prompt_input
    elif prompt_type == "Response Generation":
        st.session_state.response_generation_prompt = st.session_state.response_generation_prompt_input

############################################## AI INTERACTION FUNCTIONS ######################################################

def try_api_call(func, *args, **kwargs):
    for api_key in API_KEYS:
        for model in MODELS:
            try:
                client = OpenAI(api_key=api_key)
                return func(client, model, *args, **kwargs)
            except Exception as e:
                print(f"Error with API key {api_key[:5]}... and model {model}: {str(e)}")
    return None  # If all combinations fail

def generate_sql_query(user_input, prompt, chat_history):
    def api_call(client, model, user_input, prompt, chat_history):
        messages = [
            {"role": "system", "content": prompt},
        ]
        
        for message in chat_history:
            messages.append({"role": message["role"], "content": message["content"]})
        
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    
    return try_api_call(api_call, user_input, prompt, chat_history)

def execute_query_and_save_json(input_string, table_name, db_name=DB_NAME):
    try:
        sql_data = json.loads(input_string)
        sql_query = sql_data["SQL"]
    except json.JSONDecodeError:
        st.error("Failed to parse SQL query response.")
        return None

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
    except sqlite3.Error as e:
        st.error(f"An error occurred while executing the query: {e}")
        return None
    finally:
        conn.close()
    
    result_list = []
    for row in results:
        result_dict = {column_names[i]: row[i] for i in range(len(column_names))}
        result_list.append(result_dict)
    
    with open('query_result.json', 'w') as json_file:
        json.dump(result_list, json_file, indent=2)
    
    return result_list

def generate_response(json_data, prompt, chat_history):
    def api_call(client, model, json_data, prompt, chat_history):
        messages = [
            {"role": "system", "content": prompt},
        ]
        
        for message in chat_history:
            messages.append({"role": message["role"], "content": message["content"]})
        
        messages.append({"role": "user", "content": f"JSON data: {json_data}"})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    
    return try_api_call(api_call, json_data, prompt, chat_history)

@st.cache_data
def load_data(uploaded_file):
    try:
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        csv_analysis = analyze_csv("temp.csv")
        
        encoding = detect_encoding("temp.csv")
        df = pd.read_csv("temp.csv", encoding=encoding)
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        return None, None, None

    if not df_to_sqlite(df, FIXED_TABLE_NAME):
        return None, None, None
    
    return df, FIXED_TABLE_NAME, csv_analysis

def main():
    st.set_page_config(layout="wide", page_title="DataChat", page_icon="📈")
    st.title("Data Chat Application")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'csv_explanation' not in st.session_state:
        st.session_state.csv_explanation = ""

    if 'show_prompt_editor' not in st.session_state:
        st.session_state.show_prompt_editor = False

    df = None

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df, table_name, csv_analysis = load_data(uploaded_file)
        if df is not None:
            # Use the stored explanation as the initial value
            user_explanation = st.text_area("Please enter an explanation for your CSV data:", 
                                           value=st.session_state.csv_explanation,
                                           key="csv_explanation_input")
            
            if st.button("Submit Explanation"):
                # Update the stored explanation when the submit button is pressed
                st.session_state.csv_explanation = user_explanation
                st.success("Explanation submitted successfully!")

            # Use the stored explanation in subsequent operations
            current_explanation = st.session_state.csv_explanation if st.session_state.csv_explanation else csv_analysis
        else:
            st.warning("Failed to load the CSV file. Please try again.")
            return
    else:
        st.warning("Please upload a CSV file.")
        return

    if df is not None:
        st.sidebar.success("Data loaded successfully!")
        
        st.sidebar.subheader("Data Preview")
        st.sidebar.dataframe(df.head())

        st.header("Chat with your data")

        # Button to show/hide prompt editor
        if st.sidebar.button("Edit Prompts"):
            st.session_state.show_prompt_editor = not st.session_state.show_prompt_editor

        # Show prompt editor if the button has been clicked
        if st.session_state.show_prompt_editor:
            st.sidebar.subheader("Edit Prompts")
            
            # Add dropdown to select which prompt to edit
            prompt_type = st.sidebar.selectbox("Select prompt to edit", ["SQL Generation", "Response Generation"])

            if prompt_type == "SQL Generation":
                st.sidebar.text_area("Edit SQL Generation Prompt", value=st.session_state.sql_generation_prompt, height=300, key="sql_generation_prompt_input")
            else:
                st.sidebar.text_area("Edit Response Generation Prompt", value=st.session_state.response_generation_prompt, height=300, key="response_generation_prompt_input")

            if st.sidebar.button("Submit Prompt Changes"):
                update_prompt(prompt_type)
                st.sidebar.success(f"{prompt_type} prompt updated successfully!")

        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Create two columns for the chat input and reset button
        col1, col2 = st.columns([4, 1])

        with col1:
            prompt = st.chat_input("What would you like to know about the data?")

        with col2:
            if st.button("Reset Chat", key="reset_chat_button"):
                reset_chat()
                st.experimental_rerun()

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Generating response..."):
                sql_generation_prompt = st.session_state.sql_generation_prompt.format(csv_explanation=current_explanation, table_name=table_name)
                sql_query_response = generate_sql_query(prompt, sql_generation_prompt, st.session_state.messages[:-1])
                
                if sql_query_response is None:
                    st.error("Failed to generate a response. Please try again later.")
                else:
                    try:
                        sql_data = json.loads(sql_query_response)
                        sql_query = sql_data["SQL"]
                        display_sql_query(sql_query)
                        
                        result_list = execute_query_and_save_json(sql_query_response, table_name)

                        if result_list:
                            display_json_data(result_list)
                            
                            response_generation_prompt = st.session_state.response_generation_prompt.format(csv_explanation=current_explanation)
                            response = generate_response(json.dumps(result_list), response_generation_prompt, st.session_state.messages)

                            if response is None:
                                st.error("Failed to generate a response. Please try again later.")
                            else:
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                
                                with st.chat_message("assistant"):
                                    st.markdown(response)
                        else:
                            st.error("Failed to execute the SQL query. Please try rephrasing your question.")
                    except json.JSONDecodeError:
                        st.error("Failed to generate a valid SQL query. Please try rephrasing your question.")

if __name__ == "__main__":
    main()
