import os
import streamlit as st

apikey = st.secrets["OPENAI_API_KEY"]
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
st.title("🎥 Youtube Script Maker")
prompt = st.text_input("What do you want the video to be about?")

# Prompt templates
title_template = PromptTemplate(
    input_variables=["topic"], template="write me a youtube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables=["title", "wikipedia_research"],
    template="write me a youtube video script based on this title. Title: {title}.  You can leverage this wikipedia research: {wikipedia_research}, as long as your answer has the format of a script",
)
director_template = PromptTemplate(
    input_variables=["title", "director"],
    template="Add a very short paragraph explaining why {director} is perfect for the presenter role for the video titled {title}",
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
director_chain = LLMChain(
    llm=llm,
    prompt=director_template,
    verbose=False,
    output_key="director_script",
)
wiki = WikipediaAPIWrapper()
# show stuff if there is a prompt
if prompt:

    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    director = director_chain.run(title=title, director="Enrique")

    st.subheader(title)
    # st.write(response["title"])
    st.write(script)
    st.write(director)
    with st.expander("Title History"):
        st.info(title_memory.buffer)

    with st.expander("Script History"):
        st.info(script_memory.buffer)

    with st.expander("Wikipedia Research History"):
        st.info(wiki_research)
