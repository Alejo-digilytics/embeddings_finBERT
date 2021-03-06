import spacy
import os
from src.tools_preprocess import pdf_to_text

def NER_BERT_sm_uncased(text_name):
    # Download a spacy model for processing English
    os.system("python3 -m spacy download en_core_web_sm")
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")

    # Process a sentence using the spacy model
    local_dir_txt = os.path.join(os.getcwd(), "Data", "text")
    if ".txt" in text_name:
        pass
    else:
        text_name = text_name + ".txt"
    my_text = ""
    with open(os.path.join(local_dir_txt, text_name), "r") as file:
        for line in file:
            my_text = my_text + line
        file.close()
    doc = nlp(my_text)

    # Display the entities found by the model, and the type of each.
    print('{:<12}  {:}\n'.format('Entity', 'Type'))

    # Print each entity found
    for ent in doc.ents:
        # Print the entity text `ent.text` and its label `ent.label_`.
        print('{:<12}  {:}'.format(ent.text, ent.label_))


if __name__ == '__main__':
    pdf_to_text("DL template 01.pdf",save=True)
    NER_BERT_sm_uncased("DL template 01.txt")