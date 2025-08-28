import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import snappy
random.seed(42)
torch.manual_seed(42)

# Generate dataset
def generate_braid(num_strands=4, min_len=5, max_len=50):
    gen_range = list(range(1, num_strands)) + list(range(-num_strands+1, 0))
    length = random.randint(min_len, max_len)
    return [random.choice(gen_range) for _ in range(length)]

def is_hyperbolic_and_volume(braid):
    try:
        link = snappy.Link(braid_closure=braid)
        manifold = link.exterior()
        vol = manifold.volume()
        hyperbolic = 1 if vol > 0.1 else 0
        return hyperbolic, vol / 10.0  # Normalize volume roughly
    except:
        return 0, 0.0

dataset = []
for _ in range(3000):
    braid = generate_braid()
    label, vol = is_hyperbolic_and_volume(braid)
    dataset.append((braid, label, vol))

# Split train/val/test
train_data = dataset[:1600]
val_data = dataset[1600:2000]
test_data = dataset[2000:]

class BraidDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        braid, label, vol = self.data[idx]
        return torch.tensor(braid, dtype=torch.long), torch.tensor(label, dtype=torch.float), torch.tensor(vol, dtype=torch.float)

def collate(batch):
    braids, labels, vols = zip(*batch)
    lengths = [len(b) for b in braids]
    max_len = max(lengths) if lengths else 0
    pad_braids = torch.zeros(len(braids), max_len, dtype=torch.long)
    for i, b in enumerate(braids):
        pad_braids[i, :lengths[i]] = b
    return pad_braids, torch.tensor(labels, dtype=torch.float), torch.tensor(vols, dtype=torch.float)

train_dataset = BraidDataset(train_data)
val_dataset = BraidDataset(val_data)
test_dataset = BraidDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate)

class KnotNet(nn.Module):
    def __init__(self, num_strands=4, hidden_dim=64):
        super().__init__()
        self.num_strands = num_strands
        self.hidden_dim = hidden_dim
        self.initial_state = nn.Parameter(torch.randn(num_strands, hidden_dim))
        self.sub_thetas1 = nn.Parameter(torch.randn(1))  # For sub-braid 0-1 (pair 0)
        self.sub_thetas2 = nn.Parameter(torch.randn(2))  # For sub-braid (pairs 1,2)
        self.meta_thetas = nn.Parameter(torch.randn(3))  # For integration
        self.gates = nn.Parameter(torch.randn(3))  # Gates for pairs 0,1,2
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(num_strands * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 0: sigmoid for class, 1: linear for vol
        )

    def forward(self, braids):
        batch_size, max_len = braids.shape
        state = self.initial_state.unsqueeze(0).repeat(batch_size, 1, 1)
        gate_vals = self.gates.sigmoid()  # [3]

        for t in range(max_len):
            gen = braids[:, t]
            mask = (gen != 0)
            p = torch.abs(gen) - 1
            s = torch.sign(gen).float()

            # Sub-braid 1: pair 0 (strands 0-1)
            sub_mask1 = mask & (p == 0)
            if sub_mask1.any():
                th = self.sub_thetas1[0] * s[sub_mask1] * gate_vals[0]
                cos_th = th.cos().unsqueeze(1)
                sin_th = th.sin().unsqueeze(1)
                u = state[sub_mask1, 0].clone()
                v = state[sub_mask1, 1].clone()
                state[sub_mask1, 0] = u * cos_th - v * sin_th
                state[sub_mask1, 1] = u * sin_th + v * cos_th

            # Sub-braid 2: pairs 1 (1-2), 2 (2-3)
            for pp in [1, 2]:
                sub_mask2 = mask & (p == pp)
                if sub_mask2.any():
                    th = self.sub_thetas2[pp - 1] * s[sub_mask2] * gate_vals[pp]
                    cos_th = th.cos().unsqueeze(1)
                    sin_th = th.sin().unsqueeze(1)
                    u = state[sub_mask2, pp].clone()
                    v = state[sub_mask2, pp + 1].clone()
                    state[sub_mask2, pp] = u * cos_th - v * sin_th
                    state[sub_mask2, pp + 1] = u * sin_th + v * cos_th

        # Meta-integration: cycle through all pairs
        for pp in range(3):
            th = self.meta_thetas[pp] * gate_vals[pp]
            cos_th = th.cos()
            sin_th = th.sin()
            u = state[:, pp].clone()
            v = state[:, pp + 1].clone()
            state[:, pp] = u * cos_th - v * sin_th
            state[:, pp + 1] = u * sin_th + v * cos_th

        # Normalize per strand (out-of-place)
        state = torch.stack([self.norm(state[:, i]) for i in range(self.num_strands)], dim=1)

        flat = state.view(batch_size, -1)
        out = self.mlp(flat)
        class_out = out[:, 0].sigmoid()
        vol_out = out[:, 1]
        return class_out, vol_out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KnotNet()
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.02)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()

use_amp = device.type == 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

best_val_loss = float('inf')
patience = 10
counter = 0
for epoch in range(100):
    model.train()
    train_loss = 0
    for braids, labels, vols in train_loader:
        braids, labels, vols = braids.to(device), labels.to(device), vols.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            class_out, vol_out = model(braids)
            l = 0.7 * bce_loss(class_out, labels) + 0.3 * mse_loss(vol_out, vols)
            for name, p in model.named_parameters():
                if 'thetas' in name:
                    l += 1e-5 * p.abs().sum()
        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += l.item() * len(labels)
    train_loss /= len(train_data)
    scheduler.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for braids, labels, vols in val_loader:
            braids, labels, vols = braids.to(device), labels.to(device), vols.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                class_out, vol_out = model(braids)
            l = 0.7 * bce_loss(class_out, labels) + 0.3 * mse_loss(vol_out, vols)
            val_loss += l.item() * len(labels)
    val_loss /= len(val_data)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            break

# Test
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
correct = 0
total = 0
test_mse = 0
with torch.no_grad():
    for braids, labels, vols in test_loader:
        braids, labels, vols = braids.to(device), labels.to(device), vols.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            class_out, vol_out = model(braids)
        pred = (class_out > 0.5).float()
        correct += (pred == labels).sum().item()
        total += len(labels)
        test_mse += mse_loss(vol_out, vols).item() * len(vols)
acc = correct / total
test_mse /= total
print(f"Test classification accuracy: {acc}, Volume MSE: {test_mse}")
