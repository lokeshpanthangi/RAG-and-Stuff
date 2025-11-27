from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import File, UploadFile
import fitz # PyMuPDF
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
import os


load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")


model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)


# Pinecone

pc = Pinecone(api_key=pinecone_api_key)
index_name = "ragtest"
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)



# Prompt Template

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an AI assistant that helps people find information.
Use the following context to answer the question.
here is the context: {context}
this is the Question: {question}
Answer the question based on the context provided.""")


# Parser

parser = StrOutputParser()


# Chain 

chain = prompt | model | parser

def query_index(query: str):
    docs = vector_store.similarity_search(query, k=3)
    response = chain.invoke({"context": docs, "question": query})
    return response


def upload_file(file: UploadFile = File(...)):
    content = ""
    with fitz.open(stream=file.file.read(), filetype="pdf") as pdf:
        for page in pdf:
            content += page.get_text()



    texts = text_splitter.split_text(content)
    documents = [Document(page_content=text, metadata={"text": text}) for text in texts]
    vector_store.add_documents(documents)
    file.file.close()
    return {"filename": file.filename, "status": "uploaded"}