import os
import torch

import torch.nn as nn
import pandas as pd
import time 

from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from bm25 import BM25s
from dotenv import load_dotenv
from preload_model import model, tokenizer

load_dotenv()

TOKEN_BERT_MODEL = os.getenv("TOKEN_BERT_MODEL")
BERT_MODEL = os.getenv("BERT_MODEL")

print(torch.cuda.is_available())  
torch.cuda.set_device(0)
print(torch.cuda.current_device())

import logging
logging.disable(logging.WARNING)

os.environ["WANDB_DISABLED"] = "true"

class MultilingualBertDataset(torch.utils.data.Dataset):
   
    def __init__(self, questions, articles, labels):
        self.questions = questions
        self.articles = articles
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)
    
    def tokenize_pair_text(self, text_1, text_2):
        return self.tokenizer(text_1, text_2, padding='max_length', truncation=True)

    def __getitem__(self, index):
        encodings = self.tokenize_pair_text(self.questions[index], self.articles[index])
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item
    

# Func ----------------------------------------------------------
def min_max_scale(data):
    local_max = max(data)
    local_min = min(data) 
    output_list = []
    
    for local_data in data:
        output_list.append((local_data - local_min)/(local_max - local_min))
        
    return output_list

def get_score_predictions(dataset, model_trainner):
    score_list = model_trainner.predict(dataset).predictions
    one_label_score = softmax_model(torch.tensor(score_list)).tolist()
    output_list = []
    for score in one_label_score:
        output_list.append(score[1])
        
    return output_list

def concat_score(dataset, score_list):
    local_dataset = dataset
    local_dataset['bert_score'] = score_list
    return dataset

def min_max_scale_score(score_list, num):
    output = []
    for i in range(0, len(score_list), num):
        output += min_max_scale(score_list[i:i+num])
    return output

def model_predict(local_data, local_dataset, model_trainner, is_scale):
    bert_score_list = get_score_predictions(local_dataset, model_trainner)
    if is_scale: 
        bert_score_list = min_max_scale_score(bert_score_list, 100)
    output_dataset = concat_score(local_data, bert_score_list)
    return output_dataset

# -----------------------------------------------------------------------------
softmax_model = nn.Softmax(dim=1)

local_trainer = Trainer(
    model=model,
)

def ensemble_score(df, w_bm25, w_bert): 
    for i in range(len(df)):
        if df["bm25_score_scaled"][i] > 1:
            df["bm25_score_scaled"][i] = 1
    df["ensemble_score"] = w_bm25 * df["bm25_score_scaled"] + w_bert * df["bert_score"]
    return df


def bert_ensemble(bm25s_retriever, article_info, query):
    start_time = time.time()  
    bm25_result = BM25s(bm25s_retriever, article_info, query)
    df = pd.DataFrame(bm25_result)
    df['labels'] = 0
    private_dataset = MultilingualBertDataset(df["query"], df["content"], df["labels"])
    
    model_output = model_predict(df, private_dataset, local_trainer, True)
    model_output_ens = ensemble_score(model_output, 0.75, 0.25)
    print(len(model_output_ens)) 
    
    threshold = 0.85 
    max_ensemble_score = model_output_ens['ensemble_score'].max()
    relevant_candidates = model_output_ens[model_output_ens['ensemble_score'] >= threshold * max_ensemble_score]
    output = relevant_candidates.to_dict(orient='records')

    end_time = time.time() 
    print(f"Done BERT and Ensemble! Time taken: {end_time - start_time:.2f} seconds")

    return output