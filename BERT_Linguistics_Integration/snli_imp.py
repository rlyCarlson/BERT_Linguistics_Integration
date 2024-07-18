import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_batch(batch_index_and_data):
    batch_index, batch_data = batch_index_and_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, return_dict=True)
    model.to(device)
    model.eval()

    results = []
    pbar = tqdm(batch_data.iterrows(), total=len(batch_data), leave=False, desc=f"Batch {batch_index + 1} Processing")
    for index, row in pbar:
        input_id1 = tokenizer(row['First Sentence'], return_tensors='pt').input_ids.to(device)
        input_id2 = tokenizer(row['Second Sentence'], return_tensors='pt').input_ids.to(device)
        
        with torch.no_grad():
            output_id1 = model.generate(input_id1)
            output_id2 = model.generate(input_id2)
        
        output_text1 = tokenizer.decode(output_id1[0], skip_special_tokens=True)
        output_text2 = tokenizer.decode(output_id2[0], skip_special_tokens=True)
        results.append((output_text1, output_text2))
        pbar.set_postfix({"Processed": index})
    return results

def chunker(seq, size):
    return ((i, seq[pos:pos + size]) for i, pos in enumerate(range(0, len(seq), size)))

base_name = "t5-small"
model_name = "/Users/ishaansingh/Downloads/t5-base-finetuned-implicatures-small-3"

path = "/Users/ishaansingh/Documents/CS224N_FinalProject/snli_train_clean.csv"
snli_df = pd.read_csv(path)
snli_df['First Sentence'] = snli_df['First Sentence'].astype(str)
snli_df['Second Sentence'] = snli_df['Second Sentence'].astype(str)

if __name__ == '__main__':
    print(cpu_count())
    num_processes = min(cpu_count(), 8)
    batch_size = len(snli_df) // num_processes

    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_batch, chunker(snli_df, batch_size)), total=num_processes, desc="Overall Progress"))

    results_flat = [item for sublist in results for item in sublist]
    output_df = pd.DataFrame(results_flat, columns=['Implicature Sentence 1', 'Implicature Sentence 2'])
    csv_output_path = '/Users/ishaansingh/Documents/CS224N_FinalProject/implicatures_snli_train.csv'
    output_df.to_csv(csv_output_path, index=False)
    print(f"Implicatures saved to {csv_output_path}")
