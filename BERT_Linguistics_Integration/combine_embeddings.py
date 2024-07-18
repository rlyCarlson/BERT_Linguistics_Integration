import numpy as np
import torch
import torch.nn.functional as F

def combine_SVD(embeddings):
    # Stack the embeddings vertically to create a matrix
    # max_size = max(embeddings[0].size(0), embeddings[1].size(0))
    # new_embeddings = []
    # new_embeddings.append(F.pad(embeddings[0].unsqueeze(0), (0, max_size - embeddings[0].size(0))))
    # new_embeddings.append(F.pad(embeddings[1].unsqueeze(0), (0, max_size - embeddings[1].size(0))))
    # new_embeddings = [new_embedding.cpu().detach().numpy()
    #               for new_embedding in new_embeddings]
    # matrix = np.vstack(new_embeddings)
    #
    # # Perform SVD on the matrix
    # U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    #
    # # Select the first k left singular vectors
    # merged_embedding = U[:, :1]
    # merged_embedding = torch.from_numpy(merged_embedding)
    matrix = np.hstack([embedding.cpu().detach().numpy() for embedding in embeddings])

    # transpose the matrix
    matrix = np.expand_dims(matrix, axis=0)

    # perform SVD on the transposed matrix
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)

    # select the first k right singular vectors
    k = 1
    merged_embedding = VT[:k, :]

    # convert the merged embedding back to a PyTorch tensor
    merged_embedding = torch.from_numpy(merged_embedding).squeeze(0)

    return merged_embedding

def combine_concat(embeddings):
    return torch.cat(embeddings, dim=0)

def combine_adding(embeddings):
    # normalization
    normalized_embeddings = [(np.array(embedding.cpu()) - np.mean(np.array(embedding.cpu()))) / np.std(np.array(embedding.cpu())) for embedding in embeddings]
    summed_embedding = np.sum(normalized_embeddings, axis=0)
    return summed_embedding

if __name__ == '__main__':
    presup_file_path = 'snli_presup_test_embeddings.pt'
    naive_file_path = 'snli_test_embeddings.pt'
    svd_file_path = 'snli_test_combined_svd.pt'
    cat_file_path = 'snli_test_combined_cat.pt'
    add_file_path = 'snli_test_combined_add.pt'
    print("Starting to load embeddings")
    presup_embeddings = torch.load(presup_file_path)
    naive_embeddings = torch.load(naive_file_path)
    print(presup_embeddings[0])
    print("Done loading embeddings")
    with open(svd_file_path, 'wb') as svd_file:
        for i in range(len(presup_embeddings)):
            print(i)
            print(presup_embeddings[i].shape, naive_embeddings[i].shape)
            svd = combine_SVD([presup_embeddings[i], naive_embeddings[i]])
            print(f"Done SVD {i}", svd.shape)
            torch.save(svd, svd_file, _use_new_zipfile_serialization=False)
            
    with open(cat_file_path, 'wb') as cat_file:
        for i in range(len(presup_embeddings)):
            print(i)
            print(presup_embeddings[i].shape, naive_embeddings[i].shape)
            cat = combine_concat([presup_embeddings[i], naive_embeddings[i]])
            print(f"Done cat{i}", cat.shape)
            torch.save(cat, cat_file, _use_new_zipfile_serialization=False)
    with open(add_file_path, 'wb') as add_file:
        for i in range(len(presup_embeddings)):
            print(i)
            print(presup_embeddings[i].shape, naive_embeddings[i].shape)
            adding = combine_adding([presup_embeddings[i], naive_embeddings[i]])
            print(f"Done adding{i}", adding.shape)
            torch.save(adding, add_file, _use_new_zipfile_serialization=False)