import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm.auto import tqdm
import json
import multiprocessing

model_name = "/Users/bradleymoon/downloads/t5-base-finetuned-presuppositions-small-2"
base_name = "t5-small"
file_path = '/Users/bradleymoon/downloads/snli_1.0/snli_1.0_test.jsonl'
csv_output_path = '/Users/bradleymoon/downloads/presup_snli_test.csv'

tokenizer = AutoTokenizer.from_pretrained(base_name)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def process_line(line):
    data = json.loads(line)
    if data['gold_label'] != '-': 
        input_id1 = tokenizer(data['sentence1'], return_tensors='pt').input_ids.to(device)
        input_id2 = tokenizer(data['sentence2'], return_tensors='pt').input_ids.to(device)
        output_id1 = model.generate(input_id1)
        output_id2 = model.generate(input_id2)
        output_text1 = tokenizer.decode(output_id1[0], skip_special_tokens=True)
        output_text2 = tokenizer.decode(output_id2[0], skip_special_tokens=True)
        return (output_text1, output_text2)
    else:
        return None

def read_jsonl_file_and_generate_embeddings(file_path):
    with open(file_path, 'r') as file:
        total_lines = sum(1 for line in file)
    start_line = int(total_lines * 0.1)
    end_line = int(total_lines * 0.4)
    with open(file_path, 'r') as file:
        lines = file.readlines()[start_line:end_line]
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(process_line, lines), total=len(lines), desc="Processing SNLI Dataset"))

    outputs = [result for result in results if result is not None]
    return outputs

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)  
    outputs = read_jsonl_file_and_generate_embeddings(file_path)
    outputs_df = pd.DataFrame(outputs, columns=['presupposition_sentence1', 'presupposition_sentence2'])
    outputs_df.to_csv(csv_output_path, index=False)