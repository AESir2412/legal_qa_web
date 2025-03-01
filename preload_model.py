from transformers import BertTokenizer, BertForSequenceClassification
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN_BERT_MODEL = os.getenv("TOKEN_BERT_MODEL")
BERT_MODEL = os.getenv("BERT_MODEL")

print("🔄 Đang tải mô hình trước khi chạy Streamlit...")
model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL, 
    num_labels = 2)
tokenizer = BertTokenizer.from_pretrained(TOKEN_BERT_MODEL)
print("✅ Mô hình đã được tải!")

