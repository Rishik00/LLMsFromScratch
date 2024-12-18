import torch
import torch.nn as nn

from gpt_model import GPTModel
from load_data import get_batch, encode, decode

batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 50
eval_interval = 100
learning_rate = 1e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 0  # Update this after loading vocab
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.01

torch.manual_seed(1337)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "eval"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_gpt_model(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}")

        # sample a batch of data
        xb, yb = get_batch("train")
        xb, yb = xb.to(device), yb.to(device)

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    # Update vocab size after loading your vocabulary
    vocab_size = 123

    model = GPTModel(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
    model = model.to(device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    train_gpt_model(model)

    print("Model training complete")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
