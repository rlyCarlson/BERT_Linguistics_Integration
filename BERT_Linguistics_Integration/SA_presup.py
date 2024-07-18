import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd

model_name = "/Users/ishaansingh/Downloads/t5-base-finetuned-presuppositions-small"
base_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(base_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
train_data = dataset['test']

train_texts = train_data['text']
presuppositions = []

for text in tqdm(train_texts):
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    output_ids = model.generate(input_ids)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    presuppositions.append(output_text)

presuppositions_df = pd.DataFrame({'presupposition': presuppositions})
output_csv_path = '/Users/ishaansingh/Documents/CS224N_FinalProject/SA_presuppositions_test.csv'
presuppositions_df.to_csv(output_csv_path, index=False)
