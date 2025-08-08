from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
import re

# ---------- Config ----------
plt.rcParams['font.sans-serif'] = ['SimHei']          # display Chinese labels
plt.rcParams['axes.unicode_minus'] = False
alpha = 0.5                                           # strength of topic-level weight (0–1)

lda_results = """
主题 1: 0.010*"health" + 0.009*"comfort"  + 0.008*"human"
主题 2: 0.016*"pain" + 0.010*"illness" + 0.009*"clinical"
主题 3: 0.037*"circadian" + 0.028*"rhythm" + 0.020*"melatonin" + 0.011*"body" + 0.009*"cycle"
主题 4: 0.020*"comfort" + 0.019*"sensation"  + 0.009*"physiological"  + 0.009*"body"
主题 5: 0.014*"hospital" + 0.013*"age"  + 0.012*"stroke"  + 0.012*"disease"
主题 6: 0.016*"symptom" + 0.011*"health"
主题 7: 0.035*"exercise" + 0.009*"sweat"  + 0.008*"body"
主题 8: 0.015*"stress" + 0.014*"physiological" + 0.013*"strain"
主题 9: 0.067*"sleep"
主题 10: 0.023*"blood" + 0.013*"stress"  + 0.012*"pressure"  + 0.007*"muscle"
主题 11: 0.025*"mortality" + 0.013*"health" + 0.009*"death"
主题 12: 0.011*"performance" + 0.007*"recovery" + 0.007*"body"
主题 13: 0.050*"performance" + 0.033*"cognitive"  + 0.010*"mental"
"""

# corpus-level topic weights (from LDA)
topic_weights = np.array([
    0.1093, 0.0372, 0.0583, 0.0898, 0.0632, 0.0459,
    0.1164, 0.0619, 0.0459, 0.0661, 0.2020, 0.0761, 0.0280
], dtype=np.float32)

# ---------- Parse LDA output ----------
def parse_topic_line(line):
    return {word: float(weight) for weight, word in re.findall(r'([0-9.]+)\*"(.*?)"', line)}

topics = [parse_topic_line(l) for l in lda_results.strip().split('\n')]

# ---------- Compute topic embeddings ----------
model = SentenceTransformer('all-MiniLM-L6-v2')
topic_vecs = []

for idx, topic in enumerate(topics):
    words   = list(topic.keys())
    weights = np.array([topic[w] for w in words], dtype=np.float32)

    embeds  = model.encode(words, normalize_embeddings=True)           # (n_words, dim)
    vec     = (weights[:, None] * embeds).sum(axis=0)                  # weighted sum
    vec    /= np.linalg.norm(vec) + 1e-12                              # L2-normalize

    # append one extra dimension carrying the topic weight
    extended  = np.hstack([vec, alpha * topic_weights[idx]])
    topic_vecs.append(extended)

X = np.vstack(topic_vecs)                                              # (n_topics, dim+1)

# ---------- Hierarchical clustering ----------
dist_mat = pdist(X, metric='cosine')
linked   = linkage(dist_mat, method='average')

# ---------- Plot dendrogram ----------
plt.figure(figsize=(10, 7))
dendrogram(linked,
           labels=[f"Topic {i+1}" for i in range(len(topics))],
           distance_sort='descending')
plt.title("Hierarchical Clustering ")
plt.xlabel("Cluster")
plt.ylabel("Linkage Distance")
plt.show()
