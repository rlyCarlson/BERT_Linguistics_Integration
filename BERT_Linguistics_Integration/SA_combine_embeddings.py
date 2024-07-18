import numpy as np
import torch
import os
from tqdm import tqdm 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using MPS device:", device)

def combine_SVD(embeddings):
    matrix = np.hstack([embedding.to('cpu').detach().numpy() for embedding in embeddings])
    matrix = np.expand_dims(matrix, axis=0)
    # see paper for SVD explanation
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    merged_embedding = VT[:1, :]  
    merged_embedding = torch.from_numpy(merged_embedding).squeeze(0)
    return merged_embedding.to(device)  

def combine_concat(embeddings):
    return torch.cat(embeddings, dim=-1)

def combine_adding(embeddings):
    normalized_embeddings = [(embedding - embedding.mean()) / embedding.std() for embedding in embeddings]
    summed_embedding = sum(normalized_embeddings)
    return summed_embedding

def pad_embeddings(embeddings, max_length):
    batch_size, current_length, feature_size = embeddings.shape
    if current_length >= max_length:
        return embeddings
    padding = torch.zeros((batch_size, max_length - current_length, feature_size), device=embeddings.device)
    padded_embeddings = torch.cat([embeddings, padding], dim=1)
    return padded_embeddings

def process_directory(presup_dir, naive_dir, output_dir_svd, output_dir_cat, output_dir_add, max_length=512):
    presup_files = sorted(os.listdir(presup_dir))
    naive_files = sorted(os.listdir(naive_dir))

    os.makedirs(output_dir_svd, exist_ok=True)
    os.makedirs(output_dir_cat, exist_ok=True)
    os.makedirs(output_dir_add, exist_ok=True)

    for presup_file, naive_file in tqdm(zip(presup_files, naive_files), total=len(presup_files), desc="Processing files"):
        presup_path = os.path.join(presup_dir, presup_file)
        naive_path = os.path.join(naive_dir, naive_file)

        presup_embeddings = torch.load(presup_path, map_location=device)
        presup_embeddings = presup_embeddings[:, 0, :]
        naive_data = torch.load(naive_path, map_location=device)
        naive_embeddings, labels = naive_data['embeddings'], naive_data['labels']
        naive_embeddings = naive_embeddings[:,0, :]
        print(naive_embeddings.shape)
        if presup_embeddings.size(1) < max_length or naive_embeddings.size(1) < max_length:
            presup_embeddings = pad_embeddings(presup_embeddings, max_length)
            naive_embeddings = pad_embeddings(naive_embeddings, max_length)
        presup_embeddings_cpu = presup_embeddings.to('cpu')
        naive_embeddings_cpu = naive_embeddings.to('cpu')

        if presup_embeddings_cpu.shape[0] < naive_embeddings_cpu.shape[0]:
            padded_array = np.pad(presup_embeddings_cpu.numpy(), ((0, naive_embeddings_cpu.shape[0] - presup_embeddings_cpu.shape[0]), (0, 0)))
            presup_embeddings = torch.from_numpy(padded_array).to(device)  
        else:
            padded_array = np.pad(naive_embeddings_cpu.numpy(), ((0, presup_embeddings_cpu.shape[0] - naive_embeddings_cpu.shape[0]), (0, 0)))
            naive_embeddings = torch.from_numpy(padded_array).to(device)  
        svd_embeddings = combine_SVD([presup_embeddings, naive_embeddings])
        print(svd_embeddings.shape)
        torch.save(svd_embeddings, os.path.join(output_dir_svd, presup_file))

        cat_embeddings = combine_concat([presup_embeddings, naive_embeddings])
        print(cat_embeddings.shape)
        torch.save(cat_embeddings, os.path.join(output_dir_cat, presup_file))

        add_embeddings = combine_adding([presup_embeddings, naive_embeddings])
        print(add_embeddings.shape)
        torch.save(add_embeddings, os.path.join(output_dir_add, presup_file))

if __name__ == '__main__':
    presup_dir = '/Users/ishaansingh/Documents/CS224N_FinalProject/SA_presup_embed_train'
    naive_dir = '/Users/ishaansingh/Documents/CS224N_FinalProject/sent_train'
    output_dir_svd = '/Users/ishaansingh/Documents/CS224N_FinalProject/SA_embeddings_svd_train'
    output_dir_cat = '/Users/ishaansingh/Documents/CS224N_FinalProject/SA_embeddings_cat_train'
    output_dir_add = '/Users/ishaansingh/Documents/CS224N_FinalProject/SA_embeddings_add_train'
    max_length = 512  

    process_directory(presup_dir, naive_dir, output_dir_svd, output_dir_cat, output_dir_add, max_length)
