import spacy
from spacy.lang.en import English
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import logging

logging.basicConfig(level=logging.INFO)
spacy.prefer_gpu()


def get_tokenizer_and_model(path_to_case, output_h_s=False):
    """
    This function gets the tokenizer and the model for a given path in the library pytorch
    """
    if path_to_case == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=output_h_s)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path_to_case)
        model = AutoModel.from_pretrained(path_to_case, output_hidden_states=output_h_s)
    return model, tokenizer


def nlp_doc_obj(my_text, download=False):
    nlp = spacy.load("en_core_web_sm")
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
