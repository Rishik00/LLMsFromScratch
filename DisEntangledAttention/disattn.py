import torch
import torch.nn as nn

def delta_fn(i, j, k=3):
    relative_distance = i - j
    if relative_distance <= -k:
        return 0
    elif relative_distance >= k:
        return 2 * k - 1
    else:
        return relative_distance + k

# Input dimensions
B, T, C = 3, 8, 10  # Batch size, sequence length, input dimension
n_embed = 16  # Embedding dimension

# Sample input
x = torch.randn(B, T, C)

# Positional embedding layer (aligned with sequence length T)
pembed = nn.Embedding(T, n_embed)

# Generate positional indices (assuming positions are 0 to T-1 for each sequence)
pos_indices = torch.arange(T).unsqueeze(0).repeat(B, 1)  # Shape: (B, T)

# Content attention projection layers
K_c = nn.Linear(C, n_embed)  # Content key projection
Q_c = nn.Linear(C, n_embed)  # Content query projection
V_c = nn.Linear(C, n_embed)  # Content value projection

# Position attention projection layers
K_r = nn.Linear(n_embed, n_embed)  # Position key projection
Q_r = nn.Linear(n_embed, n_embed)  # Position query projection

# Content projections
Kc = K_c(x)  # Content keys: Shape (B, T, n_embed)
Qc = Q_c(x)  # Content queries: Shape (B, T, n_embed)
Vc = V_c(x)  # Content values: Shape (B, T, n_embed)

# Positional embeddings and projections
pos_embeds = pembed(pos_indices)  # Shape: (B, T, n_embed)
Kr = K_r(pos_embeds)  # Position keys: Shape (B, T, n_embed)
Qr = Q_r(pos_embeds)  # Position queries: Shape (B, T, n_embed)

# Content-content attention
Acc = torch.matmul(Qc, Kc.transpose(-2, -1))  # Shape: (B, T, T)

# Initialize Acp and Apc
Acp = torch.zeros_like(Acc)
Apc = torch.zeros_like(Acc)

# Compute Acp (Content-Position Attention)
Acp = torch.matmul(Qc, Kr.transpose(-2, -1))  # Shape: (B, T, T)

# Apply delta function to Acp
delta_matrix = torch.zeros((T, T), dtype=torch.long)
for i in range(T):
    for j in range(T):
        delta_matrix[i, j] = delta_fn(i, j, k=3)

delta_matrix = delta_matrix.unsqueeze(0).repeat(B, 1, 1)  # Shape: (B, T, T)
Acp = Acp.gather(2, delta_matrix)  # Apply delta function indices

# Compute Apc (Position-Content Attention)
Apc = torch.matmul(Kc, Qr.transpose(-2, -1))  # Shape: (B, T, T)

# Apply delta function to Apc
Apc = Apc.gather(1, delta_matrix.transpose(1, 2))  # Apply delta function indices

print("Acc:", Acc.shape)
print("Acp:", Acp.shape)
print("Apc:", Apc.shape)

import torch

# Combine all attention matrices
At = Acc + Acp + Apc

# Softmax across the last dimension (-1)
At = torch.nn.functional.softmax(At / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32)), dim=-1)

# Weighted sum with the value matrix
H0 = torch.matmul(At, Vc)  # Shape: (B, T, n_embed)
print("At:", At.shape)  # (B, T, T)
print("H0:", H0.shape)  # (B, T, n_embed)