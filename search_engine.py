import streamlit as st
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

wiki_api_wrapper= WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki_tool=WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

arxiv_api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv_tool=ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

search=DuckDuckGoSearchRun(name="Search")

st.title("Chat with search")
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter Groq API key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant", "content":"Hi, i am a chatbot with search capability. How can i assist you today?"}
    ]


for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if api_key:
    if prompt:=st.chat_input(placeholder="what is machine learning?") :
        st.session_state.messages.append({"role":"user", "content":prompt})
        st.chat_message("user").write(prompt)

        llm=ChatGroq(model="Llama3-8b-8192",groq_api_key=api_key,streaming=True)
        tools=[wiki_tool,arxiv_tool,search]

        search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parse_errors=True)
        
        with st.chat_message("assistant"):
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=search_agent.run(st.session_state.messages ,callbacks=[st_cb])
            st.write(response)
            st.session_state.messages.append({"role":"assistant", "content":response})

