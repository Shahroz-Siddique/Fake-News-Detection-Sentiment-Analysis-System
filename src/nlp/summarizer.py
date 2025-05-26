# src/nlp/summarizer.py
from transformers import pipeline

def summarize_text(text):
    summarizer = pipeline("summarization")
    return summarizer(text[:1024], max_length=130, min_length=30, do_sample=False)[0]['summary_text']
