import time

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import os
import pdftotext
import logging
import re

logging.basicConfig(level=logging.INFO)


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


def get_entity_indices(tokenizer, text, entity):
    """
    This function gets the index or indices of the input word within the given text.
    An entity can contain many words/tokens
    Input:
        - tokenizer (class): tokenizer depending on the model, i.e, BERTbase
        - text (str): text containing the entity to recognize the context
        - entity:
    """
    # Tokenize the entity , it may be broken into multiple tokens or sub-words.
    word_tokens = tokenizer.tokenize(entity)

    # Create a sequence of `[MASK]` tokens to put in place of `word`.
    masks_str = ' '.join(['[MASK]'] * len(word_tokens))

    # Replace the entity with the mask tokens.
    text_masked = text.replace(entity, masks_str)

    # `encode` gives back tokens, ids and adds the special tokens: [CLS] and [SEP]
    input_ids = tokenizer.encode(text_masked)

    # Use numpy's `where` function to find all indices of the [MASK] token.
    mask_token_indices = np.where(np.array(input_ids) == tokenizer.mask_token_id)[0]

    return mask_token_indices


def get_embedding(b_model, b_tokenizer, text, word=''):
    """
    Uses the provided model and tokenizer to produce an embedding for the
    provided `text`, and a "contextualized" embedding for `word`, if provided.
    RMK: The model must be output_hidden_states==True
    """
    # Check if the model gives back the hidden layers
    if b_model.config.output_hidden_states == False:
        b_model.config.output_hidden_states = True
    # If a word is provided, figure out which tokens correspond to it.
    if not word == "":
        entities_indices = get_entity_indices(b_tokenizer, text, word)

    # Encode the text, adding the (required!) special tokens, and converting to
    # PyTorch tensors. # Sentence to encode. # Add '[CLS]' and '[SEP]' # Return pytorch tensors.
    encoded_dict = b_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')

    input_ids = encoded_dict['input_ids']

    b_model.eval()

    # Run the text through the model and get the hidden states.
    bert_outputs = b_model(input_ids)

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = b_model(input_ids)
        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # because we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    # `hidden_states` has shape [13 x 1 x <sentence length> x 768]

    # Select the embeddings from the second to last layer.
    # `token_vecs` is a tensor with shape [<sent length> x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    # Convert to numpy array.
    sentence_embedding = sentence_embedding.detach().numpy()

    # If `word` was provided, compute an embedding for those tokens.
    if not word == '':
        # Take the average of the embeddings for the tokens in `word`.
        word_embedding = torch.mean(token_vecs[entities_indices], dim=0)

        # Convert to numpy array.
        word_embedding = word_embedding.detach().numpy()

        return (sentence_embedding, word_embedding)
    else:
        return sentence_embedding


def pdf_to_text(name, save=False):
    """
    Convert pdf to txt
    Input:
        - name: string with the name of the pdf, including .pdf
    Output:
        - pdftotext.PDF file
    """
    not_fail = True
    try:
        with open(os.path.join(os.getcwd(), "Data", "pdf", name), "rb") as f:
            salida = pdftotext.PDF(f)
            f.close()
    except:
        not_fail = False
    if save:
        rename = name.split(".")[0]
        output_dir = os.getcwd()
        local_dir_txt = os.path.join(output_dir, "Data", "text")
        try:
            os.mkdir(local_dir_txt)
        except FileExistsError:
            pass
        if rename + ".txt" in local_dir_txt:
            print("The txt file already exists")
        elif not_fail:
            with open(os.path.join(local_dir_txt, rename + ".txt"), "w+") as file:
                for page in salida:
                    file.write(page)
                file.close()
        else:
            print(rename + " was not downloaded")
    else:
        print("not saved")
    return


def preprocess_doc(file, min_length=1, alpha=False):
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
    my_text = ""
    with open(os.path.join(local_dir_txt, file), "r") as file:
        for line in file:
            my_text = my_text + line
        my_text = my_text.replace("\n", "  ").split("  ")
        while "" in my_text: my_text.remove("")
        last_length = 0
        aux = ""
        for sentence in my_text:
            sentence = sentence.strip()
            if len(sentence) >= 70 and sentence[-1] != ".":
                aux = aux + sentence
                last_length = 1
            if last_length == 0:
                text.append(sentence)
                last_length = 0
                aux = ""
            elif len(sentence) < 70 or sentence[-1] == ".":
                aux = aux.strip() + " " + sentence
                aux = aux.replace(". ", ".").split(".")
                [text.append(x) for x in aux if x != '']
                last_length = 0
                aux = ""
        file.close()
    # Select non-numeric values
    if alpha:
        final_text = []
        for x in text:
            if any(char.isdigit() for char in x):
                pass
            else:
                final_text.append(x)
        if final_text:
            text = final_text
        else:
            pass
    # Select values with a minimum
    if min_length == 1:
        return text
    else:
        final_text = []
        for x in text:
            if len(x) >= min_length:
                final_text.append(x)
        if final_text != []:
            return final_text
        else:
            return text


def create_embedding(file, model, tokenizer, min=1, just_alpha=False):
    pre_processed_doc = preprocess_doc(file, min_length=min, alpha=just_alpha)
    start = time.time()
    doc_emb_01 = get_embedding(model, tokenizer, pre_processed_doc, word="")
    end = time.time()
    print("It took {} to create the embedding.".format(end - start))
    return doc_emb_01
