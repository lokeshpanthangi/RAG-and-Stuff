from langchain_core.prompts import PromptTemplate
from fastapi import File, UploadFile
import fitz # PyMuPDF
from langchain_core.documents import Document
from config import model, text_splitter, vector_store, parser






# Prompt Template

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an AI assistant that helps people find information.
Use the following context to answer the question.
here is the context: {context}
this is the Question: {question}
Answer the question based on the context provided.""")



# Chain 

chain = prompt | model | parser




def query_index(query: str):
    docs = vector_store.similarity_search(query, k=10)
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