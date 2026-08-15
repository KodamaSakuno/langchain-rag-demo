import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9_]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(corpus_tokens)
        self.tfs = [Counter(doc) for doc in corpus_tokens]
        self.dls = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.dls) / max(self.n_docs, 1)
        df = Counter()
        for doc in corpus_tokens:
            df.update(set(doc))
        self.idf = {t: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        results = []
        for i, tf in enumerate(self.tfs):
            score = 0.0
            for t in tokenize(query):
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.dls[i] / self.avgdl)
                    score += self.idf[t] * f * (self.k1 + 1) / norm
            if score > 0:
                results.append((i, score))
        results.sort(key=lambda x: -x[1])
        return results[:k]
