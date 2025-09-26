# modules/qa.py
from transformers import pipeline

def load_qa_pipeline():
    QA_MODEL = "distilbert-base-uncased-distilled-squad"
    try:
        qa_pipe = pipeline("question-answering", model=QA_MODEL, tokenizer=QA_MODEL)
        return qa_pipe
    except Exception as e:
        print(f"Error loading QA model: {e}")
        return None

def answer_question(qa_pipe, question: str, context: str):
    if not qa_pipe:
        return {"answer": "QA model not loaded.", "score": 0.0}
    if not question or not context:
        return {"answer": "", "score": 0.0}
    try:
        res = qa_pipe(question=question, context=context)
        return {"answer": res.get("answer", ""), "score": float(res.get("score", 0.0))}
    except Exception as e:
        return {"answer": f"An error occurred: {e}", "score": 0.0}
