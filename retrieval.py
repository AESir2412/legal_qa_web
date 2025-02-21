import os
import json
import torch
import numpy as np  

import torch.nn as nn
import pandas as pd
import time 

from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
# from bm25 import bm25  # Importing the BM25 function
from bm25 import BM25s
from dotenv import load_dotenv

load_dotenv()

TOKEN_BERT_MODEL = os.getenv("TOKEN_BERT_MODEL")
BERT_MODEL = os.getenv("BERT_MODEL")
FINAL_K = int(os.getenv("FINAL_K"))

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
#         self.articles = self.remove_header(articles)
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(TOKEN_BERT_MODEL)

    def __len__(self):
        return len(self.labels)
    
    def remove_single_header(self, data):
        return data[data.find("]") + 1:]
    
    def remove_header(self, data):
        for i in range(len(data)):
            data[i] = self.remove_single_header(data[i])
        return data

    def tokenize_pair_text(self, text_1, text_2):
        return self.tokenizer(text_1, text_2, padding='max_length', truncation=True)

    def __getitem__(self, index):
        encodings = self.tokenize_pair_text(self.questions[index], self.articles[index])
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item
    
    def display_items(self, num_items=5):
        for i in range(min(num_items, len(self))):
            item = self[i]
            question = self.questions[i]
            article = self.articles[i]
            label = self.labels[i]
            print(f"Item {i}:\n Question: {question}\n Article: {article}\n Tokenized: {item}\n")
        if len(self) > num_items:
            print(f"...\nTotal items: {len(self)}")

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
    cnt = 0
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
model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL, 
        num_labels = 2)

softmax_model = nn.Softmax(dim=1)
def f2_metric(preds, labels, num):
    f2 = 0
    precision = 0
    recall = 0
    len_sample = 0
    for i in range(0, len(labels), num):
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        len_sample += 1
        for j in range(i, i + num):
            local_label = int(labels[j])
            local_preds = int(preds[j])
            
            if local_label == 0:
                if local_preds == 0:
                    tn += 1
                else: fp += 1
            elif local_label == 1:
                if local_preds == 1:
                    tp += 1
                else: fn += 1
        try:
            local_precision = tp / (tp + fp)
            local_recall = tp / (tp + fn)
            local_f2 = (5 * local_precision * local_recall) / (4 * local_precision + local_recall)
        except:
            local_f2 = 0
            local_precision = 0
            local_recall = 0
        f2 += local_f2
        precision += local_precision
        recall += local_recall
    return f2 / len_sample, precision/ len_sample, recall/ len_sample

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    
    # Add error handling for zero division
    try:
        f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
    except:
        f2 = 0
        
    return {
        'precision': precision,
        'recall': recall,
        'f2': f2_metric(preds, labels, 100)
    }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = softmax_model(outputs.get("logits"))
        
        loss_fct = nn.CrossEntropyLoss(weight=(torch.tensor([1.0, 3.0])).to("cuda"))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# training_args = TrainingArguments(
#     output_dir = '/kaggle/working/',
#     num_train_epochs = 5,
#     per_device_train_batch_size = 4,  
#     per_device_eval_batch_size= 1,
#     evaluation_strategy = "epoch",
#     logging_strategy="epoch",
#     save_strategy="epoch",
#     disable_tqdm = False, 
#     load_best_model_at_end=True
# )

training_args = TrainingArguments(
    output_dir = '.',
    per_device_eval_batch_size= 1,
    disable_tqdm = False
)

local_trainer = CustomTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
)

def ensemble_score(df, w_bm25, w_bert): 
    for i in range(len(df)):
        if df["bm25_score_scaled"][i] > 1:
            df["bm25_score_scaled"][i] = 1
    df["ensemble_score"] = w_bm25 * df["bm25_score_scaled"] + w_bert * df["bert_score"]
    return df

# def bert_ensemble(bm25_model, article_info, query):
#     bm25_result = bm25(bm25_model, article_info, query)
#     df = pd.DataFrame(bm25_result)
#     df['labels'] = 0
#     private_dataset = MultilingualBertDataset(df["query"], df["content"], df["labels"])
    
#     model_output = model_predict(df, private_dataset, local_trainer, True)
#     model_output_ens = ensemble_score(model_output, 0.75, 0.25)

#     model_output_ens = model_output_ens.nlargest(FINAL_K, 'ensemble_score')

#     output = model_output_ens.to_dict(orient='records')
#     # with open('model_predictions.json', 'w', encoding='utf-8') as f:
#     #     json.dump(output, f, ensure_ascii=False, indent=4)

#     print("Done BERT and Ensemble!")

#     return output


def bert_ensemble(bm25s_retriever, article_info, query):
    start_time = time.time()  
    bm25_result = BM25s(bm25s_retriever, article_info, query)
    df = pd.DataFrame(bm25_result)
    df['labels'] = 0
    private_dataset = MultilingualBertDataset(df["query"], df["content"], df["labels"])
    
    model_output = model_predict(df, private_dataset, local_trainer, True)
    model_output_ens = ensemble_score(model_output, 0.75, 0.25)
    print(len(model_output_ens)) #Fix chô nãy nữa huân ơi :< (nó luôn là 100) vì t quá lười để đọc code <33

    # model_output_ens = model_output_ens.nlargest(FINAL_K, 'ensemble_score')
    # output = model_output_ens.to_dict(orient='records')

    threshold = 0.85 #Không biết có đúng không đâu =)))))))))
    max_ensemble_score = model_output_ens['ensemble_score'].max()
    relevant_candidates = model_output_ens[model_output_ens['ensemble_score'] >= threshold * max_ensemble_score]
    output = relevant_candidates.to_dict(orient='records')

    end_time = time.time() 
    print(f"Done BERT and Ensemble! Time taken: {end_time - start_time:.2f} seconds")

    return output