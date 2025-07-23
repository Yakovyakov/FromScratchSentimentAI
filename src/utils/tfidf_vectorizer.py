import numpy as np
from collections import defaultdict

class TFIDFVectorizer:
    def __init__(self):
        self.vocab = []
        self.idf = {}

    def fit(self, documents):
        self.vocab = list(set(" ".join(documents).split()))
        N = len(documents)
        
        for word in self.vocab:
            count = sum(1 for doc in documents if word in doc.split())
            self.idf[word] = np.log(N / (1 + count))

    def transform(self, documents):
        X = np.zeros((len(documents), len(self.vocab)))
        for i, doc in enumerate(documents):
            words = doc.split()
            for word in words:
                if word in self.vocab:
                    tf = words.count(word) / len(words)
                    X[i, self.vocab.index(word)] = tf * self.idf[word]
        return X
    