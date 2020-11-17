import transformers
import os

# Hyperparameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = os.path.join(os.path.join(os.path.split(os.getcwd())[0],
                                            "Data", "FinBERT_base"), "pytorch_model.bin")
MODEL_PATH = os.path.join(os.path.join(os.path.split(os.getcwd())[0],
                                       "Data", "FinBERT_base"), "mortgage_bert.bin")
TRAINING_FILE = ""
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)
