from fastapi import FastAPI
from predict_service import find_category

app = FastAPI()


@app.get("/api/category")
async def root(title):
    return find_category(title)
