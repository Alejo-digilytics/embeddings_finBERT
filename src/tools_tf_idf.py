from collections import Counter
from spacy.lang.en import English
import numpy as np

def DF(processed_text):
    DF = {}
    for i in range(len(processed_text)):
        tokens = processed_text[i]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}
    return DF

def tf_idf(processed_text, doc_name, DF):
    """ It does not work yet !!!"""
    N = len(DF)
    tf_idf = {}
    for i in range(N):
        tokens = processed_text[i]
        counter = Counter(tokens)
        for token in np.unique(tokens):
            tf = counter[token] # / words_count
            df = DF(token)
            idf = np.log(N / (df + 1))
            tf_idf[doc_name, token] = tf * idf
    return tf_idf

