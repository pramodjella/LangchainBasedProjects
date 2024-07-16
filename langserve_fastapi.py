from langchain_groq import ChatGroq
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()



groq_api_key=os.getenv("GROQ_API_KEY")
model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system","Translate the following into {language}"),
        ("user","{text}")
    ]
)

parser=StrOutputParser()
chain=prompt_template|model|parser

app=FastAPI(title="This is langchain server", version="1.0", description="A simple API server using langchain runnable interface")

add_routes(app, chain, path="/chain")

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)