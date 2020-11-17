import os
import torch
import torch.nn as nn

# paths
output_dir = os.path.join(os.path.split(os.getcwd())[0], "Data", "FinBERT_base")
output_model_file = os.path.join(output_dir, "pytorch_model.bin")
output_config_file = os.path.join(output_dir, "config.json")

model = torch.load(output_model_file)
model = load_state_dict(torch.load(output_model_file))

class finbert(nn.Module):
    pretrained_config_archive_map = output_config_file
    model_type = "finbert"
    def __init__(self, n_input_features):
        super(finbert,self).__init__()
        self.linear = nn.Linear(n_input_features,1)


# scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")