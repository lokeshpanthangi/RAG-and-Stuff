from fastapi import FastAPI
from vector_query import query
from vector_db_injest import upload

app = FastAPI()
app.include_router(query)
app.include_router(upload)



@app.get("/health")
def health_check():
    return {"status": "healthy"}