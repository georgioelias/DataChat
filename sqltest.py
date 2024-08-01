import streamlit as st
import pandas as pd
import sqlite3
import json
from openai import OpenAI
import os
import chardet
import io

# OpenAI API setup
key=OPENAI_API_KEY
client = OpenAI(api_key=key)

def get_table_schema(table_name, db_name='data.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    schema = [{"name": column[1], "type": column[2]} for column in columns]
    conn.close()
    return schema

def df_to_sqlite(df, table_name, db_name='data.db'):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def execute_sql_to_json(query, db_name='data.db'):
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_json(orient='records')

def generate_sql_query(user_input, prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model="gpt-4",  # Use the latest available model
        messages=messages,
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def execute_query_and_save_json(input_string, table_name, db_name='data.db'):
    # Extract the SQL query
    start = input_string.find('SQL": "') + 7
    end = input_string.rfind('"')
    sql_query = input_string[start:end]
    
    # Execute the SQL query
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(sql_query)
    result = cursor.fetchone()
    conn.close()
    
    # Transform the result into a dictionary
    column_name = sql_query.split('AS ')[1].split(' FROM')[0] if ' AS ' in sql_query else 'result'
    result_dict = {column_name: result[0]}
    
    # Convert the dictionary to JSON and save to a file
    with open('query_result.json', 'w') as json_file:
        json.dump(result_dict, json_file, indent=2)
    
    return result_dict

def generate_response(json_data, prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"JSON data: {json_data}"}
    ]
    response = client.chat.completions.create(
        model="gpt-4",  # Update this to the correct model name
        messages=messages,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


@st.cache_data
def load_data(uploaded_file):
    # Read the uploaded file
    try:
        # Try reading with default UTF-8 encoding
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        # If UTF-8 fails, detect the encoding
        uploaded_file.seek(0)
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        # Try reading with detected encoding
        try:
            df = pd.read_csv(io.StringIO(raw_data.decode(encoding)))
        except UnicodeDecodeError:
            # If that fails, try common encodings
            encodings_to_try = ['iso-8859-1', 'cp1252']
            for enc in encodings_to_try:
                try:
                    df = pd.read_csv(io.StringIO(raw_data.decode(enc)))
                    break
                except UnicodeDecodeError:
                    continue
            else:
                st.error(f"Unable to read the CSV file. Please check the file encoding.")
                return None

    table_name = os.path.splitext(uploaded_file.name)[0].lower().replace(" ", "_")
    
    df_to_sqlite(df, table_name)
    return df, table_name

def main():
    st.set_page_config(layout="wide", page_title="DataChat", page_icon="📈")
    st.title("Data Chat Application")

    # Initialize session state for messages if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    df = None
    csv_explanation = ""
    table_name = ""

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df, table_name = load_data(uploaded_file)
        csv_explanation = st.text_area("Please enter an explanation for your CSV data:", 
                                       "Enter a detailed explanation of your CSV file structure here...")
        if st.button("Submit Explanation"):
            st.success("Explanation submitted successfully!")
    else:
        st.warning("Please upload a CSV file.")
        return

    if df is not None:
        st.sidebar.success("Data loaded successfully!")
        
        # Display data preview
        st.sidebar.subheader("Data Preview")
        st.sidebar.dataframe(df.head())

        # Main chat interface
        st.header("Chat with your data")

        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat input
        prompt = st.chat_input("What would you like to know about the data?")
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display the user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)

            # Show a loading spinner while generating the response
            with st.spinner("Generating response..."):
                # Generate SQL query
                sql_generation_prompt = f'''
                Table name: {table_name}
                Columns: {', '.join([col for col in df.columns])}
                {csv_explanation}

                A user will now chat with you. Your task is to transform the user's request into an SQL query that retrieves exactly what they are asking for.

                Rules:
                1. Return only two JSON variables: "Explanation" and "SQL".
                2. No matter how complex the user question is, return only one SQL query.
                3. Always return the SQL query in a one-line format.

                Example output:
                {{
                "Explanation": "The user is asking about the number of users. To retrieve this, we need to count all rows in the table.",
                "SQL": "SELECT COUNT(*) AS User_Count FROM {table_name}"
                }}

                Your prompt ends here. Everything after this is the chat with the user. Remember to always return the accurate SQL query.
                '''
                sql_query_response = generate_sql_query(prompt, sql_generation_prompt)
                
                # Execute SQL query and save as JSON
                result_dict = execute_query_and_save_json(sql_query_response, table_name)

                # Generate response
                response_generation_prompt = f'''
                Table name: {table_name}
                Columns: {', '.join([col for col in df.columns])}
                {csv_explanation}

                Now you will receive a JSON containing the SQL output that answers the user's inquiry. Your task is to use the SQL's output to answer the user's inquiry in plain English.
                '''
                response = generate_response(json.dumps(result_dict), response_generation_prompt)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display the assistant's response
            with st.chat_message("assistant"):
                st.markdown(response)
           
        # Reset button for chat (moved to the sidebar)
        st.sidebar.button("Reset Chat", on_click=reset_chat)

def reset_chat():
    st.session_state.messages = []

if __name__ == "__main__":
    main()
