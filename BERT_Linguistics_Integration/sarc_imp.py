import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm

model_name = "/Users/ishaansingh/Downloads/t5-base-finetuned-implicatures-small-3"
base_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(base_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
file_path = '/Users/ishaansingh/Documents/CS224N_FinalProject/responses_SARC.csv'
SARC_df = pd.read_csv(file_path)
SARC_df['response_text'] = SARC_df['response_text'].astype(str)

outputs = []

for text in tqdm(SARC_df['response_text'], desc='Generating Implicatures'):
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    output_ids = model.generate(input_ids)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    outputs.append(output_text)

outputs_df = pd.DataFrame({'implicature': outputs})
csv_output_path = '/Users/ishaansingh/Documents/CS224N_FinalProject/imp_SARC_train.csv'  # Added .csv to make the path a valid csv file path
outputs_df.to_csv(csv_output_path, index=False)
