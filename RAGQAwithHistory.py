import streamlit as st
from langchain.chains import create_history_aware_retriever , create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


st.title("Conversational RAG with chat history")


api_key=st.text_input("Enter your Groq API key:", type="password")

if api_key:
    llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=api_key)
    session_id=st.text_input("Session ID", value="default_session")
    
    if "store" not in st.session_state:
        st.session_state.store={}
    
    uploaded_files=st.file_uploader("Upload pdfs to ask questions: ", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temp_pdf=uploaded_file.name
            with open(temp_pdf,"wb") as file:
                file.write(uploaded_file.getvalue())
            
            loader=PyPDFLoader(temp_pdf)
            docs=loader.load()
            documents.extend(docs)
        
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever= vectorstore.as_retriever()

        contextualized_system_prompt=(
            "rewrite a stand alone question from given question and chat history."
            "do not answer the question"
        )

        contextualized_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualized_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualized_prompt)

        system_prompt=("you are helpful assistant. Answer user question based on given context."
                       "Context: {context}")
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}",)
            ]
        )
        qa_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,qa_chain)


        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        conversation_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )


        user_input=st.text_input("Enter your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversation_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.success("Answer:\n"+response['answer'])
            st.write("Chat history:", session_history.messages)