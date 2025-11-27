from fastapi import FastAPI
from rag_with_ragas.rag import query_index_with_eval
from reranker.query import query_cohere_index
from basic_rag.rag import query_index, upload_file
from rag_text_tables.rag import upload_file2
from rag_text_tables_images.rag import upload_file_multimodal
from fastapi import File, UploadFile


app = FastAPI()


@app.get("/query/")
async def query(question: str):
    answer = query_index(question)
    return {"question": question, "answer": answer}


@app.get("/query_cohere_reranker/")
async def query_cohere(question: str):
    answer = query_cohere_index(question)
    return {"question": question, "answer": answer}

@app.get("/query_with_evaluation/")
async def query_with_evaluation(question: str):
    result = query_index_with_eval(question)
    return {"question": question, "answer": result["response"], "ragas_scores": result["ragas_scores"]}






@app.post("/uploadfile/")
async def upload(file: UploadFile = File(...)):
    upload_file(file)
    return {"filename": file.filename, "status": "uploaded"}


@app.post("/uploadfile_with_tables/")
async def upload(file: UploadFile = File(...)):
    upload_file2(file)
    return {"filename": file.filename, "status": "uploaded"}


@app.post("/uploadfile_multimodal/")
async def upload_multimodal(file: UploadFile = File(...)):
    upload_file_multimodal(file)
    return {"filename": file.filename, "status": "uploaded"}

