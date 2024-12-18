import os
import requests
import tiktoken
import numpy as np

def load_data(
        file_name: str = 'input.txt', 
        data_url: str = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    ):
    input_file_path = os.path.join(os.path.dirname(__file__), file_name)
    if not os.path.exists(input_file_path):
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)

    return "Success", input_file_path

def tokenize_data():
    status, file_path = load_data()
    print(f"Status for loadiing data: {tokenize_data}")

    with open (file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    enc = tiktoken.get_encoding('gpt2')
    train_ids, val_ids = enc.encode_ordinary(train_data), enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    return "Done exporting and tokenizing files"


if __name__ == "__main__":
    tokenize_data()