import torch

# Parameters
batch_size = 4  # How many independent sequences to process in parallel
block_size = 8  # Maximum context length for predictions
vocab_size = 65

# Load data
file_path = r'C:\Users\sridh\OneDrive\Desktop\webdev\GPTFromScratch\data\input.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# Unique characters in the dataset
chars = sorted(list(set(data)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # String -> List of integers
decode = lambda l: ''.join([itos[i] for i in l])  # List of integers -> String

# Encode data and split into train/val
encoded_data = encode(data)
split_index = int(0.9 * len(encoded_data))

def split(data, split_index):
    train_data, val_data = data[:split_index], data[split_index:]
    return train_data, val_data

# Batch generation
def get_batch(split_type):
    train_data, val_data = split(encoded_data, split_index)
    d = train_data if split_type == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(d[i:i + block_size]) for i in ix])
    y = torch.stack([torch.tensor(d[i + 1:i + block_size + 1]) for i in ix])
    return x, y

# Main code
if __name__ == "__main__":
    xb, yb = get_batch('train')
    print('Inputs:')
    print(xb.shape)
    print(xb)
    print('Targets:')
    print(yb.shape)
    print(yb)

    print('----')

    for b in range(batch_size):  # Batch dimension
        for t in range(block_size):  # Time dimension
            context = xb[b, :t + 1]
            target = yb[b, t]
            print(f"when input is {context.tolist()} the target: {target}")
