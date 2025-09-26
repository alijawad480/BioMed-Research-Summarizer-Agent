import streamlit as st
import tempfile, os, requests
from dotenv import load_dotenv

# Import your modules
from modules.pdf_reader import extract_text_from_pdf
from modules.summarizer import summarize_text, load_summarizer_pipeline
from modules.qa import load_qa_pipeline, answer_question
from modules.trend_analysis import analyze_trends, plot_trend_counts
from modules.novelty import novelty_score
from modules.paper_search import fetch_paper_by_title

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 🎨 Streamlit page config
st.set_page_config(
    page_title="BioMed Research Summarizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for theme
st.markdown("""
<style>
    /* Background and font */
    body {
        background-color: #f8f9fc;
        font-family: 'Segoe UI', sans-serif;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #222831;
        color: white;
    }
    [data-testid="stSidebar"] h2 {
        color: #00adb5;
    }
    /* Cards */
    .stCard {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 BioMed Research Summarizer Agent")
st.write("Your AI-powered assistant for summarization, Q&A, novelty detection, and trend analysis of biomedical papers.")

# Load models once
@st.cache_resource
def get_summarizer_pipeline():
    return load_summarizer_pipeline()
summarizer_pipeline = get_summarizer_pipeline()

@st.cache_resource
def get_qa_pipeline():
    return load_qa_pipeline()
qa_pipeline = get_qa_pipeline()

if not summarizer_pipeline:
    st.error("⚠️ Summarization model failed to load. Check Hugging Face token or internet.")
if not qa_pipeline:
    st.error("⚠️ QA model failed to load. Try smaller model or increase RAM.")

# Sidebar Navigation
st.sidebar.title("📌 Menu")
mode = st.sidebar.radio(
    "Select a Feature",
    ["📄 Upload PDF", "🔍 Search by Title", "❓ Q&A", "📊 Trend Analysis", "✨ Novelty Detection", "🌐 HTTP Agent Info"]
)

# ---------------------- Features ----------------------

if mode == "📄 Upload PDF":
    st.subheader("📄 Upload a Research Paper (PDF)")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("⏳ Extracting text..."):
            tempf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tempf.write(uploaded_file.read())
            tempf.close()
            with open(tempf.name, "rb") as fh:
                text = extract_text_from_pdf(fh)
            os.unlink(tempf.name)

        if text.strip():
            st.success("✅ Text extracted successfully!")
            summary_output = summarize_text(summarizer_pipeline, text)
            with st.expander("📌 Summarized Text", expanded=True):
                st.write(summary_output)
        else:
            st.error("❌ Could not extract text. Try another PDF.")

elif mode == "🔍 Search by Title":
    st.subheader("🔍 Search Paper by Title (Semantic Scholar)")
    title_q = st.text_input("Enter paper title or keywords")
    if st.button("Search & Summarize"):
        with st.spinner("🔎 Searching..."):
            results = fetch_paper_by_title(title_q, limit=5)

        if not results:
            st.warning("⚠️ No results found.")
        else:
            options = [f"{p.get('title','')} ({p.get('year','')})" for p in results]
            choice = st.selectbox("Select a paper", options)
            idx = options.index(choice)
            p = results[idx]

            st.markdown(f"**📝 Title:** {p.get('title')}")
            st.markdown(f"**📅 Year:** {p.get('year')}")
            st.markdown(f"**👥 Authors:** {p.get('authors')}")

            if st.button("📥 Download & Summarize"):
                st.info("⏳ Downloading PDF...")
                try:
                    r = requests.get(p["openAccessPdf"]["url"], stream=True, timeout=20)
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    with open(tmp.name, "wb") as fh:
                        for chunk in r.iter_content(1024*32):
                            fh.write(chunk)

                    with open(tmp.name, "rb") as fh:
                        text = extract_text_from_pdf(fh)
                    os.unlink(tmp.name)

                    st.success("✅ PDF downloaded & summarized")
                    st.write(summarize_text(summarizer_pipeline, text))

                except Exception as e:
                    st.error(f"❌ Error: {e}")

elif mode == "❓ Q&A":
    st.subheader("❓ Ask Questions from Research Papers")
    context = st.text_area("Paste paper text or summary", height=250)
    question = st.text_input("Ask your question:")
    if st.button("Get Answer"):
        res = answer_question(qa_pipeline, question, context)
        st.success("✅ Answer Found")
        st.write("**Answer:**", res.get("answer", ""))
        st.write("**Confidence:**", f"{res.get('score', 0.0):.2f}")

elif mode == "📊 Trend Analysis":
    st.subheader("📊 Trend Analysis of Abstracts")
    text = st.text_area("Paste multiple abstracts", height=250)
    top_n = st.slider("Number of keywords", 5, 50, 15)
    if st.button("Analyze Trends"):
        common = analyze_trends(text, top_n=top_n)
        st.bar_chart({w: c for w, c in common})

elif mode == "✨ Novelty Detection":
    st.subheader("✨ Novelty Detection")
    target = st.text_area("Target paper abstract", height=150)
    related_bulk = st.text_area("Related abstracts (one per line)", height=150)
    if st.button("Compute Novelty"):
        related_texts = [r.strip() for r in related_bulk.splitlines() if r.strip()]
        nov = novelty_score(target, related_texts)
        st.metric("Novelty Score", f"{nov['score']:.2f}")
        st.write("🔑 Most novel sentences:")
        for d in sorted(nov["details"], key=lambda x: x["novelty"], reverse=True)[:5]:
            st.markdown(f"- {d['sent']}  \n*(Novelty: {d['novelty']:.2f})*")

elif mode == "🌐 HTTP Agent Info":
    st.subheader("🌐 HTTP Agent Info")
    st.code("""
GET /health
POST /summarize (multipart file OR JSON {"title": "..."})
    """)

# Footer
st.markdown("---")
st.caption("🚀 BioMed Research Summarizer – Hackathon Prototype")
