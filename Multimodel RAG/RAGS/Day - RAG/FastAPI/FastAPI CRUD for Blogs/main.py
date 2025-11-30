from fastapi import FastAPI
from routes import router
from database import engine, Base


Base.metadata.create_all(bind=engine)


app = FastAPI()
app.include_router(router)

# Serve the HTML page at the root URL

