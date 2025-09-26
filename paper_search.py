# modules/paper_search.py
import requests

SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_PAPER = "https://api.semanticscholar.org/graph/v1/paper/"

def fetch_paper_by_title(title, limit=3):
    params = {
        "query": title,
        "limit": limit,
        "fields": "title,abstract,year,openAccessPdf,url,authors"
    }
    try:
        r = requests.get(SEMANTIC_SCHOLAR_SEARCH, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
        return data
    except Exception as e:
        return []
