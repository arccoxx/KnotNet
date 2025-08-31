import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import time
import matplotlib.pyplot as plt

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# --- Synthetic Dataset ---
# This class generates synthetic braid data to avoid dependencies like snappy.
class SyntheticBraidDataset(Dataset):
    def __init__(self, size=240, max_len=20):
        self.data = []
        for _ in range(size):
            braid = [random.randint(-3, 3) for _ in range(random.randint(5, max_len))]
            label = random.choice([0.0, 1.0])
            vol = random.uniform(0, 1)
            self.data.append((braid, label, vol))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        braid, label, vol = self.data[idx]
        return torch.tensor(braid, dtype=torch.long), torch.tensor(label, dtype=torch.float), torch.tensor(vol, dtype=torch.float)

# Collate function to pad braids in a batch
def collate(batch):
    braids, labels, vols = zip(*batch)
    lengths = [len(b) for b in braids]
    max_len = max(lengths) if lengths else 0
    pad_braids = torch.zeros(len(braids), max_len, dtype=torch.long)
    for i, b in enumerate(braids):
        pad_braids[i, :lengths[i]] = b
    return pad_braids, torch.tensor(labels, dtype=torch.float), torch.tensor(vols, dtype=torch.float)

# Create datasets and loaders with small sizes for quick testing
train_dataset = SyntheticBraidDataset(size=240)
val_dataset = SyntheticBraidDataset(size=30)
test_dataset = SyntheticBraidDataset(size=30)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate)

# --- Basic KnotNet (Recurrent v2 Baseline with Multilayer) ---
# This is a recurrent model inspired by early KnotNet v2, with stacked layers for multilayer processing.
class KnotNet(nn.Module):
    def __init__(self, num_strands=4, hidden_dim=64, num_layers=2):
        super().__init__()
        self.num_strands = num_strands
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.initial_state = nn.Parameter(torch.randn(num_strands, hidden_dim))
        self.thetas = nn.Parameter(torch.randn(num_layers, 3))  # Trainable angles per layer, per pair
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.mlp = nn.Sequential(
            nn.Linear(num_strands * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, braids):
        batch_size, max_len = braids.shape
        state = self.initial_state.unsqueeze(0).repeat(batch_size, 1, 1)
        for layer in range(self.num_layers):
            for t in range(max_len):
                gen = braids[:, t]
                mask = (gen != 0)
                p = torch.abs(gen) - 1
                s = torch.sign(gen).float()
                for pp in range(3):
                    sub_mask = mask & (p == pp)
                    if sub_mask.any().item():
                        th = self.thetas[layer, pp] * s[sub_mask]
                        cos_th = th.cos().unsqueeze(1)
                        sin_th = th.sin().unsqueeze(1)
                        u = state[sub_mask, pp].clone()
                        v = state[sub_mask, pp + 1].clone()
                        state[sub_mask, pp] = u * cos_th - v * sin_th
                        state[sub_mask, pp + 1] = u * sin_th + v * cos_th
            state = self.norms[layer](state)
        flat = state.view(batch_size, -1)
        out = self.mlp(flat)
        return out[:, 0].sigmoid(), out[:, 1]

# --- Transformer KnotNet (Standard Transformer Baseline with Multilayer) ---
# This uses a standard Transformer encoder for parallel processing of braid sequences.
class TransformerKnotNet(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(7, hidden_dim)  # Embedding for generators -3 to 3
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, braids):
        embedded = self.embedding((braids.clamp(-3, 3) + 3))
        transformed = self.transformer_encoder(embedded)
        state = transformed.mean(dim=1)  # Aggregate over sequence
        out = self.mlp(state)
        return out[:, 0].sigmoid(), out[:, 1]

# --- KnotHyperTransformer (Custom Hypergraph Transformer with Multilayer) ---
# This is a custom hypergraph Transformer integrated with knot theory, using bipartite attention for nodes (strands) and hyperedges (grouped crossings).
class KnotHyperTransformerLayer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_strands=4):
        super().__init__()
        self.num_strands = num_strands
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.rotation = nn.Parameter(torch.randn(num_strands-1))  # Knot rotation parameters
        self.norm = nn.LayerNorm(d_model)

    def forward(self, combined):
        total_seq = combined.shape[1]
        attn_mask = torch.full((total_seq, total_seq), float('-inf'), device=combined.device)
        
        # Correctly define the bipartite attention mask
        num_strands = self.num_strands
        attn_mask[:num_strands, num_strands:] = 0  # Nodes attend to hyperedges
        attn_mask[num_strands:, :num_strands] = 0  # Hyperedges attend to nodes

        attn_out, _ = self.self_attn(combined, combined, combined, attn_mask=attn_mask)
        rot_th = self.rotation.mean()
        cos_th, sin_th = rot_th.cos(), rot_th.sin()
        attn_out = attn_out * cos_th - attn_out.roll(1, dims=1) * sin_th  # Apply knot rotation
        return self.norm(attn_out)

class KnotHyperTransformer(nn.Module):
    def __init__(self, hidden_dim=32, stride=3, num_layers=2, num_strands=4, m_precision=10):
        super().__init__()
        self.num_strands = num_strands
        self.embedding = nn.Embedding(7, hidden_dim)
        self.layers = nn.ModuleList([KnotHyperTransformerLayer(hidden_dim, nhead=4, num_strands=num_strands) for _ in range(num_layers)])
        self.stride = stride
        self.framings = nn.Parameter(torch.randn(num_strands-1))
        self.m = m_precision
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, braids):
        batch_size, seq_len = braids.shape
        embedded = self.embedding((braids.clamp(-3, 3) + 3))
        nodes = embedded.mean(dim=1).unsqueeze(1).repeat(1, self.num_strands, 1)
        num_hyper = max(1, (seq_len + self.stride - 1) // self.stride)
        hyperedges = torch.zeros(batch_size, num_hyper, embedded.shape[-1]).to(embedded.device)
        for i in range(num_hyper):
            start = i * self.stride
            end = min(start + self.stride, seq_len)
            chunk = embedded[:, start:end].mean(dim=1)
            frame_th = self.framings.mean() * torch.pi + (i / self.m)
            cos_f, sin_f = frame_th.cos(), frame_th.sin()
            chunk = chunk * cos_f - chunk.roll(1) * sin_f
            hyperedges[:, i] = chunk
        combined = torch.cat((nodes, hyperedges), dim=1)
        for layer in self.layers:
            combined = layer(combined)
        state = combined.mean(dim=1)
        out = self.mlp(state)
        return out[:, 0].sigmoid(), out[:, 1]

# --- Training and Evaluation Function ---
def train_and_eval(model, epochs=150, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    max_memory = 0
    train_start = time.time()
    for epoch in range(epochs):
        model.train()
        for braids, labels, vols in train_loader:
            braids, labels, vols = braids.to(device), labels.to(device), vols.to(device)
            optimizer.zero_grad()
            class_out, vol_out = model(braids)
            l = 0.7 * bce_loss(class_out, labels) + 0.3 * mse_loss(vol_out, vols)
            l.backward()
            optimizer.step()
            max_memory = max(max_memory, torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)

    train_time = time.time() - train_start

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for braids, labels, vols in val_loader:
            braids, labels, vols = braids.to(device), labels.to(device), vols.to(device)
            class_out, vol_out = model(braids)
            l = 0.7 * bce_loss(class_out, labels) + 0.3 * mse_loss(vol_out, vols)
            val_loss += l.item() * len(labels)
    val_loss /= len(val_loader.dataset)

    return train_time, val_loss, max_memory

# --- Benchmark ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define number of layers for all models
NUM_LAYERS = 16
print(f"--- BENCHMARKING WITH {NUM_LAYERS} LAYERS ---")

baseline_model = KnotNet(num_layers=NUM_LAYERS)
baseline_train_time, baseline_val_loss, baseline_memory = train_and_eval(baseline_model, device=device)
print(f"KnotNet v2: Train time {baseline_train_time:.4f}s, Val loss {baseline_val_loss:.4f}, Max memory {baseline_memory / 1e6:.2f} MB")

transformer_model = TransformerKnotNet(num_layers=NUM_LAYERS)
transformer_train_time, transformer_val_loss, transformer_memory = train_and_eval(transformer_model, device=device)
print(f"Transformer: Train time {transformer_train_time:.4f}s, Val loss {transformer_val_loss:.4f}, Max memory {transformer_memory / 1e6:.2f} MB")

# --- HyperTransformer Stride Analysis ---
hyper_strides = [1, 2, 3, 4, 5]
hyper_val_losses = []
hyper_train_times = []
print("\n--- BENCHMARKING KnotHyperTransformer with varying strides ---")

for stride in hyper_strides:
    hyper_model = KnotHyperTransformer(stride=stride, num_layers=NUM_LAYERS)
    train_time, val_loss, memory = train_and_eval(hyper_model, device=device)
    print(f"KnotHyper (stride={stride}): Train time {train_time:.4f}s, Val loss {val_loss:.4f}, Max memory {memory / 1e6:.2f} MB")
    hyper_val_losses.append(val_loss)
    hyper_train_times.append(train_time)
    
# --- Plotting Results ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot Validation Loss vs. Stride
ax1.plot(hyper_strides, hyper_val_losses, marker='o', linestyle='-', color='b')
ax1.set_title('KnotHyperTransformer Performance vs. Stride')
ax1.set_xlabel('Stride')
ax1.set_ylabel('Validation Loss')
ax1.grid(True)
ax1.set_xticks(hyper_strides)

# Plot Training Time vs. Stride
ax2.plot(hyper_strides, hyper_train_times, marker='o', linestyle='-', color='r')
ax2.set_xlabel('Stride')
ax2.set_ylabel('Training Time (s)')
ax2.grid(True)
ax2.set_xticks(hyper_strides)

plt.tight_layout()
plt.show()

