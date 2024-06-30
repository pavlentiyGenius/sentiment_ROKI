import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizerFast

MODEL_CHECKPOINT = "Blaxzter/LaBSE-sentence-embeddings"
NUM_LABELS = 3
BATCH_SIZE = 32
# DEVICE = 'cpu'

MAPPING = {0:'негатив', 1:'нейтрально', 2:'позитив'}

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_CHECKPOINT)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, NUM_LABELS))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs['last_hidden_state'][:, 0, :]
        x = self.classifier(x)
        return x
    
class SenitmentTorch:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
        self.max_length = 512
        

    def prediction(self, text):
        encoding = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        input_ids = nn.functional.pad(input_ids, (0, self.max_length - input_ids.shape[0]), value=0)
        attention_mask = nn.functional.pad(attention_mask, (0, self.max_length - attention_mask.shape[0]), value=0)
        
        all_preds = []
        all_probs = []
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            input_ids = input_ids.resize_(1, self.max_length)
            attention_mask = attention_mask.resize_(1, self.max_length)
            
            outputs = self.model(input_ids, attention_mask)
            _, predictions = torch.max(outputs, 1)

            all_probs.append(np.array(outputs.cpu()))
            all_preds.append(predictions.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        all_preds = np.vectorize(MAPPING.get)(all_preds)
        return all_preds, all_probs