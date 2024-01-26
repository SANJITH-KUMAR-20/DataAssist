import os
from apikey import key
import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent, python
from langchain_experimental.tools import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.llms import openai
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
            # duplicates = pandas_agent.run("Is there any duplicates in the data...? If yes give some statistics, Give more suggestions on how to deal with them")
            # st.write(duplicates)
            # st.write("**Data Summarization**")
            # st.write(df.describe())
            # corr_analysis = pandas_agent.run('''Is there any significant correlation between the columns of the data
            #                                 ?, give the correlation analysis of the data.''')
            # st.write(corr_analysis)
            # outliers_analysis = pandas_agent.run('''Is there any outliers that i should pay attention to...? If yes... give some suggestions''')
            # st.write(outliers_analysis)
            return


        @st.cache_data
        def func_ques_var():
            st.line_chart(df, y = [user_ques])
            summary = pandas_agent.run(f"Give me a summary of the statistics of {user_ques}")
            st.write(summary)
            return

        @st.cache_data
        def function_ques_data():
            info = pandas_agent.run(extra_ques)
            st.write(info)
            return
        
        @st.cache_resource
        def wiki(prompt):
            wiki = WikipediaAPIWrapper().run(prompt) 
            return wiki
        
        @st.cache_data
        def prompt_Template():
            template = PromptTemplate(
                        input_variables=["business_problem"],
                        template = "Convert the following business problem into a data science problem : {buiness_problem}"
                    )
            template2 = PromptTemplate(
                        input_variables=["Ml_problem"],
                        template = "Give a list of machine learning algorithms for this data problem : {ML_problem}"
                    )
            return template, template2
        @st.cache_data
        def chains():
            data_problem_chain = LLMChain(llm = llm, prompt = prompt_Template()[0], verbose=True, output_key="data_prob")
            model_chain = LLMChain(llm = llm, prompt = prompt_Template()[1], verbose=True, output_key="model_selection")
            seq_chain = SequentialChain(chains= [data_problem_chain, model_chain], verbose = True, input_variables=["business_problem"],output_key=["data_prob","model_selection"])
            return seq_chain

        @st.cache_data
        def chains_out(prompt, wiki):
            chain = chains()
            chains_output = chain({"business_problem": prompt, "wikipedia_research": wiki})
            data_prob = chains_output["data_prob"]
            mod = chains_output["model_selection"]
            return data_prob, mod
        
        @st.cache_data
        def list_to_selection(my_selection_input):
            algorithms = my_selection_input.split("\n")
            algorithm = [algo.split(":")[-1].split('.')[-1].strip() for algo in algorithms if algo.strip()]
            algorithm.insert(0, "Select Algorithm")
            format_list = [f"{alg}" for alg in algorithm if alg]
            return format_list

        with st.sidebar:
            with st.expander("Choose from this"):
                st.write(steps_eda())

        @st.cache_data
        def python_agent():
            agent_exec = create_python_agent(
                llm = llm,
                tool = PythonREPLTool(),
                verbose = True,
                agent_type = AgentType.ZERO_SHOT_RECT_DESCRIPTION,
                handle_parsing_errors = True
            )
            return agent_exec
        
        @st.cache_data
        def python_solution(problem, selected_alg, csv):
            solution = python_agent().run(f"Write a python code to solve this {problem} using this algorithm{selected_alg} useg this dataset {csv}")
            return solution

        function_agent()

        st.subheader("Variable of study")
        user_ques = st.text_input("what is your variable of interest")

        func_ques_var()

        if user_ques:
            extra_ques = st.text_input("Is there you wish to ask about the data..?")
            if extra_ques == "No":
                st.write("")
            if extra_ques and user_ques and extra_ques != "No":
                function_ques_data()
            if extra_ques:
                st.divider()
                st.header("Data Science Problem")
                st.write("Now that we have a solid grasp of data at hand and a clear understanding of how data works... let us frame this into a data science problem")

                prompt = st.text_area("Add your business problem...")
                
                
                if prompt:
                    wiki_rep = wiki(prompt)
                    out = chains_out(prompt, wiki_rep)
                    data_prob = out[0]
                    mod = out[1]
                    st.write(data_prob)
                    st.write(out)

                    format_i =list_to_selection(mod)
                    selected_alg = st.selectbox("Select Machine leaning algorithm", format_i)
                    if selected_alg:
                        st.subheader("solution")
                        solution = python_solution(data_prob, selected_alg, csv)
                        st.write(solution)

            