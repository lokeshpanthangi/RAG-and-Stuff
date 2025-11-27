from config import model,parser, vector_store
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import os



compressor = CohereRerank(
    cohere_api_key=os.getenv("COHERE_API_KEY"), 
    model="rerank-english-v3.0",  
    top_n=5                       # Final number of chunks you want to get from the total retrived chunks to send to GPT
)


base_retriever = vector_store.as_retriever(search_kwargs={"k": 25})


compressed_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

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

def query_cohere_index(query: str):
    ranked_docs = compressed_retriever.invoke(query)
    normal_docs = vector_store.similarity_search(query, k=10)
    print(f"Normal Docs Retrieved: {normal_docs}")
    print(f"Ranked Docs Retrieved: {ranked_docs}")
    response = chain.invoke({"context": ranked_docs, "question": query})
    return response