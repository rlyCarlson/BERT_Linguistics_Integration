import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm.auto import tqdm
import json

model_name = "/Users/ishaansingh/Downloads/t5-base-finetuned-presuppositions-2"
base_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(base_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def read_jsonl_file_and_generate_embeddings(file_path):
    with open(file_path, 'r') as file:
        total_lines = sum(1 for line in file)

    outputs = []
    with open(file_path, 'r') as file:
        for line in tqdm(file, total=total_lines, desc="Processing SNLI Dataset"):
            data = json.loads(line)
            if data['gold_label'] != '-':
                input_id1, input_id2 = tokenizer(data['sentence1'], return_tensors='pt').input_ids, tokenizer(data['sentence2'], return_tensors='pt').input_ids
    
                output_id1, output_id2 = model.generate(input_id1), model.generate(input_id2)
    
                output_text = tokenizer.decode(output_id1[0], skip_special_tokens=True), tokenizer.decode(output_id2[0], skip_special_tokens=True)
                outputs.append(output_text)
    return outputs

file_path = '/Users/ishaansingh/Downloads/snli_1.0/snli_1.0_train.jsonl'
outputs = read_jsonl_file_and_generate_embeddings(file_path)
outputs_df = pd.DataFrame({'presuppositions': outputs})
csv_output_path = '/Users/ishaansingh/Documents/CS224N_FinalProject/presup_snli_train'
outputs_df.to_csv(csv_output_path, index=False)