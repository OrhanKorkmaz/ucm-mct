"""Değerlendirme metrikleri: retrieval, kosinüs, geometri korunumu,
k-NN tutarlılığı, analoji (vektör aritmetiği)."""
import numpy as np
from scipy.stats import spearmanr


def _norm(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def retrieval(pred, pool, true_idx, ks=(1, 5, 20)):
    """pred [n, d] çevrilen vektörler; pool [N, d] hedef uzay havuzu;
    true_idx [n] doğru havuz indeksi. Zor koşul: doğru vektör havuzda."""
    P, Q = _norm(pred), _norm(pool)
    sims = P @ Q.T
    ranks = (sims > sims[np.arange(len(P)), true_idx][:, None]).sum(1)
    return {f"top{k}": float((ranks < k).mean()) for k in ks}


def mean_cosine(pred, target):
    return float((_norm(pred) * _norm(target)).sum(-1).mean())


def geometry_spearman(source, pred, n_sample=500, seed=0):
    """Kaynak uzaydaki ikili kosinüs yapısı çeviri sonrası korunuyor mu?"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(source), min(n_sample, len(source)), replace=False)
    S, P = _norm(source[idx]), _norm(pred[idx])
    iu = np.triu_indices(len(idx), k=1)
    rho = spearmanr((S @ S.T)[iu], (P @ P.T)[iu]).correlation
    return float(rho)


def knn_consistency(pred, target, k=10):
    """Çevrilen vektörlerin hedef uzaydaki k-NN kümesi, gerçek hedef
    vektörlerin k-NN kümesiyle ne kadar örtüşüyor (Jaccard)."""
    P, T = _norm(pred), _norm(target)
    simP, simT = P @ T.T, T @ T.T
    np.fill_diagonal(simP, -np.inf)
    np.fill_diagonal(simT, -np.inf)
    nnP = np.argsort(-simP, axis=1)[:, :k]
    nnT = np.argsort(-simT, axis=1)[:, :k]
    jac = [len(set(a) & set(b)) / len(set(a) | set(b))
           for a, b in zip(nnP, nnT)]
    return float(np.mean(jac))


ANALOGIES = [  # (a, b, c, d):  a - b + c ≈ d
    ("king", "man", "woman", "queen"),
    ("paris", "france", "italy", "rome"),
    ("big", "bigger", "small", "smaller"),
    ("walking", "walk", "swim", "swimming"),
    ("boy", "girl", "brother", "sister"),
    ("day", "night", "sun", "moon"),
    ("good", "better", "bad", "worse"),
    ("cat", "cats", "dog", "dogs"),
    ("france", "french", "england", "english"),
    ("father", "mother", "son", "daughter"),
    ("go", "went", "come", "came"),
    ("high", "low", "hot", "cold"),
]


def analogy_score(vectors_by_word, pool_words, pool_vecs, ks=(1, 5)):
    """Çevrilen uzayda a-b+c'nin en yakını d mi? (a,b,c havuzdan çıkarılır)"""
    Q = _norm(pool_vecs)
    word2i = {w: i for i, w in enumerate(pool_words)}
    hits = {k: 0 for k in ks}
    n = 0
    for a, b, c, d in ANALOGIES:
        if not all(w in vectors_by_word for w in (a, b, c)) or d not in word2i:
            continue
        q = vectors_by_word[a] - vectors_by_word[b] + vectors_by_word[c]
        sims = _norm(q[None]) @ Q.T
        for w in (a, b, c):
            if w in word2i:
                sims[0, word2i[w]] = -np.inf
        order = np.argsort(-sims[0])
        rank = int(np.where(order == word2i[d])[0][0])
        for k in ks:
            hits[k] += rank < k
        n += 1
    return {f"analogy_top{k}": hits[k] / n for k in ks} | {"n_analogies": n}
