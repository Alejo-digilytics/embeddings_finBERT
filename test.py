import spacy

from src.tools import sentence_tokenization, nlp_doc_obj
from src.tools_preprocess import word_spliter, raw_text_extracter, lower_case

global nlp
nlp = spacy.load("en_core_web_sm")
spacy.prefer_gpu()


raw_text = raw_text_extracter("JH Template 1.pdf")
text_in_words = word_spliter(raw_text,"list")
text_in_words_STR = word_spliter(raw_text,"STRING")
print("--------------------- WORDS -----------------------")
for w in text_in_words:
    print(w)
text_in_sentences = sentence_tokenization(raw_text)
print("--------------------- SENTENCES -----------------------")
text_in_sentences = lower_case(text_in_sentences)
for sent in text_in_sentences:
    print(sent)
nlp_doc_obj(text_in_words_STR, download=False)