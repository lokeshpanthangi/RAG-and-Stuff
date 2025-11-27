from fastapi import FastAPI
from rag import query_index, upload_file
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