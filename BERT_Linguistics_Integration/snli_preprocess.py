import pandas as pd
import json
from tqdm.auto import tqdm


def clean_and_store_sentences(file_path, csv_output_path):
    sentences1 = []
    sentences2 = []
    
    with open(file_path, 'r') as file:
        total_lines = sum(1 for line in file)
    
    with open(file_path, 'r') as file:
        for line in tqdm(file, total=total_lines, desc="Processing SNLI Dataset"):
            data = json.loads(line)
            if data['gold_label'] != '-':
                sentences1.append(data['sentence1'])
                sentences2.append(data['sentence2'])
    
    data = {'First Sentence': sentences1, 'Second Sentence': sentences2}
    df = pd.DataFrame(data)
    
    df.to_csv(csv_output_path, index=False)
    print(f"Data saved to {csv_output_path}")

file_path = '/Users/ishaansingh/Downloads/snli_1.0/snli_1.0_test.jsonl'
csv_output_path = '/Users/ishaansingh/Documents/CS224N_FinalProject/snli_test_clean.csv'

clean_and_store_sentences(file_path, csv_output_path)
