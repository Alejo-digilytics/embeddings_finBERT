from NER_test import NER_BERT_sm_uncased
from tools import get_embedding, get_tokenizer_and_model, preprocess_doc, create_embedding
from scipy.spatial.distance import cosine
import logging
import time

logging.basicConfig(level=logging.INFO)

cosine_sim = lambda x, y: 1 - cosine(x, y)

if __name__ == '__main__':
    BERT_model, BERT_tokenizer = get_tokenizer_and_model("bert-base-uncased", output_h_s=True)
    Acc_cert_1 = create_embedding("Account_certificate_1.pdf", BERT_model, BERT_tokenizer, min=1, just_alpha=True)
    bill_1 = create_embedding("utils_bill_1.pdf", BERT_model, BERT_tokenizer, min=1, just_alpha=True)
    doc_emb_JH_01 = create_embedding("JH Template 1.pdf", BERT_model, BERT_tokenizer, min=1, just_alpha=True)
    doc_emb_JH_02 = create_embedding("JH Template 2.pdf", BERT_model, BERT_tokenizer, min=1, just_alpha=True)
    doc_emb_DL_01 = create_embedding("DL template 01.txt", BERT_model, BERT_tokenizer, min=1, just_alpha=True)
    doc_emb_DL_02 = create_embedding("DL template 02.txt", BERT_model, BERT_tokenizer, min=1, just_alpha=True)
    print("_______________________")
    difference_1 = cosine_sim(doc_emb_DL_02, doc_emb_JH_01)
    print("Same kind of document and different bank (DL-JH):")
    print(difference_1, "\n")
    difference_2 = cosine_sim(doc_emb_JH_02, doc_emb_DL_01)
    print("Same kind of document and different bank (DL-JH):")
    print(difference_2, "\n")
    print("_______________________")
    print("Same kind of document and bank (DL):")
    difference_same_1 = cosine_sim(doc_emb_DL_01, doc_emb_DL_02)
    print(difference_same_1, "\n")
    print("Same kind of document and bank (JH):")
    difference_same_JH = cosine_sim(doc_emb_JH_01, doc_emb_JH_02)
    print(difference_same_JH, "\n")
    print("_______________________")
    print("Different kind of document (Bank statement, account certificate):")
    difference_same_JH = cosine_sim(doc_emb_JH_01, Acc_cert_1)
    print(difference_same_JH, "\n")
    print("Different kind of document (Bank statement, account certificate):")
    difference_docs = cosine_sim(doc_emb_DL_02, Acc_cert_1)
    print(difference_docs, "\n")
    print("Different kind of document (Bank statement, bill):")
    difference_docs = cosine_sim(doc_emb_DL_02, bill_1)
    print(difference_docs, "\n")
