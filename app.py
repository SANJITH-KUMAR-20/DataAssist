import os
from apikey import key
import streamlit as st
import pandas as pd

from langchain.llms import openai
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv


#OpenAIEnv
os.environ['OPENAI_API_KEY'] = key
load_dotenv(find_dotenv())

#model

llm = openai.OpenAI()

#Ui
st.title("DataAssist")

st.write("Welcome..! I am your AI assistant and I will be here with you and will assist you in AI projects")


with st.sidebar:
    st.write('*Your data analysis begins with csv file*')
    
    st.caption('''**Your Analysis and Machine learning Journey
             starts with uploading a csv file. Once we have your data we will preprocess it and present to you
             the most wonderful data analysis experience beyond your wildest desires**''')
    
    
    st.divider()

    st.caption("<p style = 'text-align:center'> assist </p>",unsafe_allow_html=True)
#pandas questions

if "clicked" not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    st.header("You are now ready for your machine learning experience")

    csv = st.file_uploader("upload your file here", type = "csv")


    @st.cache_data
    def steps_eda():
        return llm("What are the steps for EDA")
    
    if csv:
        csv.seek(0)
        df = pd.DataFrame(csv)

        llm = openai.OpenAI()

        @st.cache_data
        def steps_eda():
            return llm("What are the steps for EDA")

        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True)

        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first few rows of your dataset looks like this.")
            st.write(df.head())
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run('''Elaborate on the columns, and give description for each column
                                        What is the datatype of each column''')
            st.write(columns_df)
            missing_values = pandas_agent.run('''Is there any missing values in each column...?, If yes give the
                                            statistics as the number of missing values in each columns and provide suggestions on how to remove them
                                            or impute them ?''')
            st.write(missing_values)
            duplicates = pandas_agent.run("Is there any duplicates in the data...? If yes give some statistics, Give more suggestions on how to deal with them")
            st.write(duplicates)
            st.write("**Data Summarization**")
            st.write(df.describe())
            corr_analysis = pandas_agent.run('''Is there any significant correlation between the columns of the data
                                            ?, give the correlation analysis of the data.''')
            st.write(corr_analysis)
            outliers_analysis = pandas_agent.run('''Is there any outliers that i should pay attention to...? If yes... give some suggestions''')
            st.write(outliers_analysis)
            return


        @st.cache_data
        def func_ques_var():
            st.line_chart(df, y = [user_ques])
            summary = pandas_agent.run(f"Give me a summary of the statistics of {user_ques}")
            st.write(summary)
            return

    

    with st.sidebar:
        with st.expander("Choose from this"):
            st.write(steps_eda())


    function_agent()

    st.subheader("Variable of study")
    user_ques = st.text_input("what is your variable of interest")

