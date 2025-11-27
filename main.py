from fastapi import FastAPI
from basic_rag.rag import query_index, upload_file
from rag_text_tables.rag import upload_file2
from fastapi import File, UploadFile


app = FastAPI()



@app.post("/uploadfile/")
async def upload(file: UploadFile = File(...)):
    upload_file(file)
    return {"filename": file.filename, "status": "uploaded"}


@app.get("/query/")
async def query(question: str):
    answer = query_index(question)
    return {"question": question, "answer": answer}


@app.post("/uploadfile_with_tables/")
async def upload(file: UploadFile = File(...)):
    upload_file2(file)
    return {"filename": file.filename, "status": "uploaded"}