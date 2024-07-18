import numpy as np
import torch
import os
from tqdm import tqdm  

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using MPS device:", device)

def combine_SVD(embeddings):
    matrix = np.hstack([embedding.to('cpu').detach().numpy() for embedding in embeddings])
    matrix = np.expand_dims(matrix, axis=0)

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
    presup_files = [os.path.join(presup_dir, file_name) for file_name in sorted(os.listdir(presup_dir))]
    naive_files = [os.path.join(naive_dir, file_name) for file_name in sorted(os.listdir(naive_dir))]

    os.makedirs(output_dir_svd, exist_ok=True)
    os.makedirs(output_dir_cat, exist_ok=True)
    os.makedirs(output_dir_add, exist_ok=True)

    for i, (presup_file, naive_file) in enumerate(tqdm(zip(presup_files, naive_files), total=len(presup_files), desc="Processing files")):
        print(presup_file)
        presup_data = torch.load(presup_file + '/embeddings.pt', map_location=device)
        naive_data = torch.load(naive_file + '/embeddings.pt', map_location=device)

        presup_embeddings = presup_data['features']
        naive_embeddings, labels = naive_data['features'], naive_data['labels']
        
        presup_embeddings_cpu = presup_embeddings.to('cpu')
        naive_embeddings_cpu = naive_embeddings.to('cpu')
        if presup_embeddings_cpu.shape[0] < naive_embeddings_cpu.shape[0]:
            padded_array = np.pad(presup_embeddings_cpu.numpy(), ((0, naive_embeddings_cpu.shape[0] - presup_embeddings_cpu.shape[0]), (0, 0)))
            presup_embeddings = torch.from_numpy(padded_array).to(device)  # Convert back to tensor and move to original device
        else:
            padded_array = np.pad(naive_embeddings_cpu.numpy(), ((0, presup_embeddings_cpu.shape[0] - naive_embeddings_cpu.shape[0]), (0, 0)))
            naive_embeddings = torch.from_numpy(padded_array).to(device)  # Convert back to tensor and move to original device
        svd_embeddings = combine_SVD([presup_embeddings, naive_embeddings])
        cat_embeddings = combine_concat([presup_embeddings, naive_embeddings])
        add_embeddings = combine_adding([presup_embeddings, naive_embeddings])
        file_name = 'embeddings.pt'  # This gets just the 'embeddings.pt' part
        svd_path = os.path.join(output_dir_svd, f'Batch_{i + 1}')
        concat_path = os.path.join(output_dir_cat, f'Batch_{i + 1}')
        add_path = os.path.join(output_dir_add, f'Batch_{i + 1}')
        os.makedirs(svd_path, exist_ok=True)
        os.makedirs(concat_path, exist_ok=True)
        os.makedirs(add_path, exist_ok=True)
        torch.save({'embeddings': svd_embeddings, 'labels': labels}, os.path.join(svd_path, file_name))
        torch.save({'embeddings': cat_embeddings, 'labels': labels}, os.path.join(concat_path, file_name))
        torch.save({'embeddings': add_embeddings, 'labels': labels}, os.path.join(add_path, file_name))

if __name__ == '__main__':
    presup_dir = '/Users/ishaansingh/Documents/CS224N_FinalProject/SARC_presup_Embeddings_train'
    naive_dir = '/Users/ishaansingh/Documents/CS224N_FinalProject/SARC_Embeddings'
    output_dir_svd = '/Users/ishaansingh/Documents/CS224N_FinalProject/SARC_presup_embeddings_svd_train'
    output_dir_cat = '/Users/ishaansingh/Documents/CS224N_FinalProject/SARC_presup_embeddings_cat_train'
    output_dir_add = '/Users/ishaansingh/Documents/CS224N_FinalProject/SARC_presup_embeddings_add_train'
    max_length = 512  

    process_directory(presup_dir, naive_dir, output_dir_svd, output_dir_cat, output_dir_add, max_length)
