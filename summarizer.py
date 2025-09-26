import os
from transformers import pipeline
from dotenv import load_dotenv

# Load token from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def load_summarizer_pipeline():
    try:
        model_name = "facebook/bart-large-cnn"
        summarizer_pipeline = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            use_auth_token=HF_TOKEN if HF_TOKEN else None  # optional
        )
        return summarizer_pipeline
    except Exception as e:
        print(f"Error loading summarizer model: {e}")
        return None

def chunk_text(text, max_tokens=800):
    """Text ko chhote parts (chunks) me todta hai."""
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_text(summarizer_pipeline, text, max_len=200):
    if not summarizer_pipeline:
        return "Summarization model not loaded."
    if len(text.strip()) == 0:
        return "No text to summarize."

    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer_pipeline(
                chunk,
                max_length=max_len,
                min_length=30,
                do_sample=False
            )
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            summaries.append(f"[Error summarizing chunk: {e}]")

    return " ".join(summaries)
