import os
import numpy as np
try:
    import pdftotext
except:
    pass
import pandas as pd
from sklearn import preprocessing as prep


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


def lower_case(data):
    return np.char.lower(data)


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


def preprocess_doc_emb(file, min_length=1, alpha=False):
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