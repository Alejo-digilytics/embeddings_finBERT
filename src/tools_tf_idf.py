from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import numpy as np
from src.tools import pdf_to_text
import os
import spacy

global nlp
nlp = spacy.load("en_core_web_sm")
spacy.prefer_gpu()


def raw_text_extracter(file):
    # Check file format and convert to txt
    if ".pdf" in file:
        pdf_to_text(file, save=True)
        file = file.replace(".pdf", ".txt")
    elif ".txt" in file:
        pass
    else:
        file = file + ".txt"
        # Fix the path of the file
    local_dir_txt = os.path.join(os.getcwd(), "Data", "text")
    text = []
    my_text = text = ""
    with open(os.path.join(local_dir_txt, file), "r") as file:
        for line in file:
            my_text = my_text + line
        file.close()
    return my_text

def word_spliter(my_text,list_or_text = "list"):
    symbols = "!\"#$&()*+-.,/:;<=>?@[\]^_`{|}~\n–"
    for i in symbols:
        my_text = my_text.replace(i, ' ')
    text = my_text.split()
    if list_or_text == "list":
        return text
    else:
        return " ".join(text)



def nlp_doc_obj(my_text,download=False):
    doc = nlp(my_text)

    # Create list of word tokens
    token_list = []
    for token in doc:
        token_list.append(token.text)
    print(token_list)
    return doc

def sentence_tokenization(text):
    # Load English tokenizer, tagger, parser, NER and word vectors
    nlp = English()

    # Create the pipeline 'sentencizer' component
    sbd = nlp.create_pipe('sentencizer')

    # Add the component to the pipeline
    nlp.add_pipe(sbd)

    #  "nlp" Object is used to create documents with linguistic annotations.
    doc = nlp(text)

    # create list of sentence tokens
    sents_list = []
    for sent in doc.sents:
        sents_list.append(sent.text)
    return sents_list

def lower_case(data):
    return np.char.lower(data)

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
    N = len(DF)
    tf_idf = {}
    for i in range(N):
        tokens = processed_text[i]
        counter = Counter(tokens)
        for token in np.unique(tokens):
            tf = counter[token] / words_count
            df = DF(token)
            idf = np.log(N / (df + 1))
            tf_idf[doc_name, token] = tf * idf
    return tf_idf

