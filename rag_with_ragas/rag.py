from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy,LLMContextPrecisionWithoutReference
from datasets import Dataset 
from langchain_core.prompts import PromptTemplate
from config import model, vector_store, parser,embeddings


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


def query_index_with_eval(query: str):
    # 1. Retrieve Docs
    docs = vector_store.similarity_search(query, k=3) # Retrieve top 3
    
    # 2. Extract content from docs for RAGAS
    retrieved_contexts = [doc.page_content for doc in docs]
    
    # 3. Generate Answer
    response = chain.invoke({"context": docs, "question": query})
    
    # 4. Prepare Data for RAGAS (It expects a specific dataset format)
    data = {
        'question': [query],
        'answer': [response],
        'contexts': [retrieved_contexts], # Must be a list of lists
    }
    
    dataset = Dataset.from_dict(data)
    
    # 5. Run Evaluation (Reference-Free)
    # Note: We omit 'context_recall' because it needs ground truth

    context_precision = LLMContextPrecisionWithoutReference()
    results = evaluate(
        dataset=dataset, 
        metrics=[
            faithfulness, 
            answer_relevancy,
            context_precision
        ],
        llm=model,
        embeddings=embeddings
    )
    
    return {
        "response": response,
        "ragas_scores": results
    }