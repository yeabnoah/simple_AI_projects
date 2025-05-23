from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer("Hugging Face Transformers is a library for NLP and more. It gives you easy access to powerful models.", max_length=30, min_length=5, do_sample=False)

print(summary)