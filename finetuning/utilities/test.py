import torch



# Sample data as a PyTorch tensor
data_t = torch.tensor([
    [3, 2, 1, 4, 0],  # 0 represents 'A'
    [5, 3, 2, 1, 1],  # 1 represents 'B'
    [2, 1, 0, 3, 0],  # 0 represents 'A'
    [4, 2, 1, 0, 1],  # 1 represents 'B'
    [6, 3, 2, 1, 2],   # 2 represents 'C'
    [6, 3, 2, 1, 2]
], dtype=torch.float32)

def get_val_by_treshold(data):
    # Extract the labels (last column)
    t = data[:, -1]

    # Find unique labels
    unique_t = t.unique()

    # Create a tensor to store the result for each unique t
    acc_t = torch.zeros_like(unique_t, dtype=torch.float32)

    # Iterate over each unique t, perform the calculation, and store the result
    for i, label in enumerate(unique_t):
        # Select rows where the label matches the current label and columns 0-3
        d = data[t == label, :4]
        
        # Perform the calculation
        acc = (d[:, 0] + d[:, 1]) / (d[:, 0] + d[:, 1] + d[:, 2] + d[:, 3])
        # Store the mean result in acc_t
        acc_t[i] = acc.mean()
        
    return acc_t


print(get_val_by_treshold(data_t))

print(get_val_by_treshold(data_t))