from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os 

load_dotenv()

def get_rag_chain() :
    
    vectordb = Chroma(persist_directory="db/", embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    rag = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever,
        return_source_documents=True 
    )

    return rag