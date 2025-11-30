from fastapi import FastAPI
from docs import docs_router

app = FastAPI()
app.include_router(docs_router)


@app.get("/health")
def read_health():
    return {"status": "healthy"}
