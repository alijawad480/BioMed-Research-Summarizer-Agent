# modules/novelty.py
from sentence_transformers import SentenceTransformer, util
import re
import numpy as np

EMB_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight

def sentences_from_text(text):
    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 30]
    return sents

def novelty_score(target_text, related_texts):
    """
    Compute novelty score between 0..1 (1 = more novel)
    related_texts: list of strings
    """
    if not target_text:
        return {"score": 0.0, "details": []}
    target_sents = sentences_from_text(target_text)
    related_sents = []
    for r in related_texts:
        related_sents.extend(sentences_from_text(r))
    if not related_sents:
        return {"score": 1.0, "details": [{"sent": s, "max_sim": 0.0, "novelty": 1.0} for s in target_sents]}

    emb_t = EMB_MODEL.encode(target_sents, convert_to_tensor=True)
    emb_r = EMB_MODEL.encode(related_sents, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(emb_t, emb_r)  # tensor
    import torch
    max_sims, _ = torch.max(sims, dim=1)
    max_sims = max_sims.cpu().numpy()
    nov_scores = (1.0 - max_sims).tolist()
    overall = float(np.mean(nov_scores)) if len(nov_scores) > 0 else 0.0
    details = [{"sent": target_sents[i], "max_sim": float(max_sims[i]), "novelty": nov_scores[i]} for i in range(len(target_sents))]
    return {"score": overall, "details": details}
