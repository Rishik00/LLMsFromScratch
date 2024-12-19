import torch
import torch.nn as nn
from torch.nn import functional as F    
import matplotlib.pyplot as plt

# Local imports
from utils import decode, encode, get_batch
torch.manual_seed(1337)

batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
vocab_size = 65

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx, targets=None):
        logits = self.embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def train_model(model, epochs=10000, batch_size=32, lr=1e-3):
    # Create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # List to store loss values for plotting
    loss_values = []
    
    for step in range(epochs):  # Increase number of steps for good results...
        # Sample a batch of data
        xb, yb = get_batch('train')
        
        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Store loss for visualization
        loss_values.append(loss.item())
        
        # Print loss occasionally for monitoring
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item()}")

    # Plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label="Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model



if __name__ == "__main__":
    xb, yb = get_batch('train')
    m = BigramLanguageModel(vocab_size)
    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss)

    print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

    m=train_model(m)

    print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
