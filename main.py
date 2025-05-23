from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

class SummaryRequest(BaseModel):
    text: str
    max_length: int = 30
    min_length: int = 5
    do_sample: bool = False

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.post("/summarize")
async def summarize(req: SummaryRequest):
    input_text = req.text
    summary = summarizer(input_text, max_length=120, min_length=30, do_sample=False)
    return {"summary": summary[0]['summary_text']}

