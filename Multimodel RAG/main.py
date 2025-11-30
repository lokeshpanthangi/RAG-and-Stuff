from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_docling.loader import DoclingLoader
from langchain_core.prompts import PromptTemplate
from fastapi import FastAPI, File, UploadFile
from dotenv import load_dotenv
from openai import chat
from pinecone import Pinecone
import tempfile
import os



# ... Initialization ...

load_dotenv()
app = FastAPI()

# ... Pinecone ...
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("startnew")

# ... OpenAI Embeddings and ChatBot ...
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)
model = ChatOpenAI(model="gpt-4o-mini")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)



parser = StrOutputParser()

prompt = PromptTemplate(
    template="Given the context: {context}, answer the question: {question} and never mention about the context , and also analyze the context Before answering and Answer to the point if there are no relavent chunks for the question, you can use Your Knowledge",
    input_variables=["context", "question"]
)

# ... Chain ...
chain = prompt | model | parser

#Routes
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Save uploaded file to a temp file
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    # Use the temp file path with DoclingLoader
    file_loader = DoclingLoader(tmp_path)
    documents = file_loader.load()

    # Clean metadata to ensure Pinecone compatibility
    for doc in documents:
        # Remove complex metadata that Pinecone can't handle
        if 'dl_meta' in doc.metadata:
            del doc.metadata['dl_meta']
        # Keep only simple metadata values (strings, numbers, booleans)
        clean_metadata = {}
        for key, value in doc.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                clean_metadata[key] = value
        doc.metadata = clean_metadata

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(documents)
    # Upsert documents into Pinecone
    vector_store.add_documents(documents)
    # Clean up temp file
    os.unlink(tmp_path)

    return {"content": documents[0].page_content if documents else ""}


chat_history = []
@app.post("/query/")
async def query_documents(query: str):
    global chat_history
    chat_history.append({"role": "user", "content": query})
    # Get relevant documents from Pinecone
    relevant_docs = vector_store.similarity_search(query)
    # Generate a response using the chain
    response = chain.invoke({"context": relevant_docs,"question": chat_history})
    chat_history.append({"role": "assistant", "content": response})
    return {"response": response}


@app.get("/show_history")
async def show_history():
    global chat_history
    return {"chat_history": chat_history}