import sys 
sys.path.insert(0, '../../')
import random
from common.common import create_folder
from common.pytorch import load_model
import pytorch_pretrained_bert as Bert
from model.utils import age_vocab
from common.common import load_obj
from dataLoader.MLM import MLMLoader
from torch.utils.data import DataLoader
import pandas as pd
import pickle as pkl
import argparse
from model.MLM import BertForMaskedLM
from model.optimiser import adam
import sklearn.metrics as skm
import numpy as np
import torch
import time
import torch.nn as nn
import os


def make_example_survival_data():
    """
    Generate synthetic example survival data in a format compatible with BEHRT.
    
    - caliber_id is structured as: CLS -> visit 1 tokens -> SEP -> visit 2 tokens -> SEP -> etc
    - More 'I10' codes increase risk of DEATH (shorter target_time).
    - Other random codes lead to other target_event outcomes.
    
    Returns:
        pd.DataFrame with columns: patid, caliber_id, age, target_event, target_time
    """

	os.makedirs("local_example", exist_ok=True)

    def make(n, seed=42):
        random.seed(seed)
        np.random.seed(seed)
    
        possible_codes = ["I10", "E11", "E78", "J45", "E23", "I101"]
        non_death_targets = ["E11", "E78", "J45", "E23", "I101"]
    
        rows = []
        for i in range(n):
            patid = f"P{i:04d}"
            age0 = np.random.randint(20, 80)  # baseline age
    
            # Random number of tokens in each segment
            n_tokens_before = np.random.randint(1, 4)
            n_tokens_after = np.random.randint(1, 4)
    
            # Build tokens
            tokens_before = random.choices(possible_codes, k=n_tokens_before)
            tokens_after = random.choices(possible_codes, k=n_tokens_after)
    
            # Insert SEP separators
            caliber_id = ["CLS"] + tokens_before + ["SEP"] + tokens_after + ["SEP"]
            ages = [age0] * (len(tokens_before) + 2) + [age0 + 1] * (len(tokens_after) + 1)
    
            # Count I10s
            n_i10 = caliber_id.count("I10")
    
            # Determine target event and time
            if n_i10 > 0:
                target_event = "DEATH"
                # More I10s leds to shorter survival time
                base_time = np.random.uniform(0.95, 1.05)
                target_time = max(0.1, base_time / (1 + n_i10))
            else:
                target_event = random.choice(non_death_targets)
                target_time = np.random.uniform(0.1, 1.0)
    
            rows.append(
                {
                    "patid": patid,
                    "caliber_id": caliber_id,
                    "age": ages,
                    "target_event": target_event,
                    "target_time": round(target_time, 2),
                }
            )
    
        return pd.DataFrame(rows)

    df = make(n=10000)
    df.to_parquet("local_example/data_train.parquet", index=False)

    df = make(n=1000)
    df.to_parquet("local_example/data_val.parquet", index=False)

    df = make(n=1000)
    df.to_parquet("local_example/data_test.parquet", index=False)
	
    token2idx = {
	    'PAD': 0,
	    'UNK': 1,
	    'SEP': 2,
	    'CLS': 3,
	    'MASK': 4,
	    'I10': 5,
        'I101': 6,
	    'E11': 7,
        'E23': 8,
	    'E78': 9,
	    'J45': 10,
        'DEATH': 11
	}	

	# Wrap in a dict with 'token2idx' key
    bert_vocab = {'token2idx': token2idx}
		
    with open("local_example/token2idx.pkl", "wb") as f:
        pkl.dump(bert_vocab, f)

if __name__ == "__main__":
	
	make_example_survival_data()
