import time
import numpy as np
import torch
from src.tools_preprocess import preprocess_doc_emb


def create_embedding(file, model, tokenizer, min=1, just_alpha=False):
    pre_processed_doc = preprocess_doc_emb(file, min_length=min, alpha=just_alpha)
    start = time.time()
    doc_emb_01 = get_embedding(model, tokenizer, pre_processed_doc, word="")
    end = time.time()
    print("It took {} to create the embedding.".format(end - start))
    return doc_emb_01


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