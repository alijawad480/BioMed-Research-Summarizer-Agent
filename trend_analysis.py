# modules/trend_analysis.py
import re
from collections import Counter
import matplotlib.pyplot as plt

def analyze_trends(text, top_n=15):
    if not text:
        return []
    words = re.findall(r'\b[a-zA-Z\-]{3,}\b', text.lower())
    # filter out common words - simple stoplist
    stop = set(["the","and","for","with","that","this","are","was","were","from","have","has","but","not","using","use","using"])
    words = [w for w in words if w not in stop]
    common = Counter(words).most_common(top_n)
    return common

def plot_trend_counts(common_list):
    if not common_list:
        return None
    labels = [w for w, _ in common_list]
    counts = [c for _, c in common_list]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(labels[::-1], counts[::-1])
    ax.set_xlabel("Frequency")
    ax.set_title("Top keywords")
    plt.tight_layout()
    return fig
