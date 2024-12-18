import torch
import torch.nn as nn

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 50
eval_interval = 100
learning_rate = 1e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 0
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.01

torch.manual_seed(1337)

class LayerNorm1D(nn.Module):
    def __init__(self, dim, eps=0.5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.beta = torch.zeros(dim)
        self.gamma = torch.ones(dim)
    
    def __call__(self, x):
        # calculate the forward pass
        xmean = x.mean(1, keepdim=True) # batch mean
        xvar = x.var(1, keepdim=True) # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)  # Added the missing dropout definition
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        attn_wei = q @ k.transpose(-2, -1)  # (B, T, T)
        attn_wei = attn_wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn_wei = torch.softmax(attn_wei, dim=-1)  # Fixed `torch.F.softmax` to `torch.softmax`
        attn_wei = self.dropout(attn_wei)

        v = self.value(x)
        out = attn_wei @ v  # (B, T, head_size)

        return out 


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(AttentionHead(head_size=head_size) for _ in range(num_heads))
        self.proj_layer = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate outputs of all heads
        out = self.dropout(self.proj_layer(out))  # Fixed input to `proj_layer` from `x` to concatenated `out`
        return out
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.feedforward_net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.feedforward_net(x)


class GPTBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForwardNetwork(n_embed=n_embed)
        self.lnorm1 = nn.LayerNorm(n_embd)
        self.lnorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections with layer norm
        x = x + self.sa(self.lnorm1(x))
        x = x + self.ffwd(self.lnorm2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[GPTBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        tok_embeddings = self.token_embedding(idx)
        pos_embeddings = self.positional_embedding(torch.arange(idx.size(1), device=idx.device))
        x = tok_embeddings + pos_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Truncate context dynamically
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Focus on last token
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # Append next token
        return idx
