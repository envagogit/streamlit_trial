import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

##### REMEMBER TO SAVEEE! #####

os.environ[
    "OPENAI_API_KEY"
] = apikey  # Setting a dictionary so that we can send the API Key to OpenAI

# pip install streamlit langchain openai wikipedia chromadb tiktoken

# App framework
st.title("😃 langChain")
prompt = st.text_input("Plug your prompt here:")

# Prompt templates
title_template = PromptTemplate(
    input_variables=["topic"], template="write me a youtube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables=["title", "wikipedia_research"],
    template="write me a youtube video script based on this title. Title: {title} while leveraging this wikipedia research: {wikipedia_research}",
)
# Memory
title_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
script_memory = ConversationBufferMemory(input_key="title", memory_key="chat_history")
# Llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(
    llm=llm,
    prompt=title_template,
    verbose=True,
    output_key="title",
    memory=title_memory,
)
script_chain = LLMChain(
    llm=llm,
    prompt=script_template,
    verbose=True,
    output_key="script",
    memory=script_memory,
)
""" SEQUENTIAL CHAIN
sequential_chain = SequentialChain(
    chains=[title_chain, script_chain],
    input_variables=["topic"],
    output_variables=["title", "script"],
    verbose=False,
)
"""
wiki = WikipediaAPIWrapper()
# show stuff if there is a prompt
if prompt:
    """SEQUENTIAL CHAIN
    response = sequential_chain(
        {"topic": prompt}
    )
    # With normal sequential chains you need to pass the inputs as a dictionary and not use "run"
    """
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.subheader(title)
    # st.write(response["title"])
    st.write(script)

    with st.expander("Title History"):
        st.info(title_memory.buffer)

    with st.expander("Script History"):
        st.info(script_memory.buffer)

    with st.expander("Wikipedia Research History"):
        st.info(wiki_research)
