"""
Fixed KnotNet Benchmarking Suite
Properly handles SnapPy Number objects and avoids slow census operations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import snappy
import time
import json
import random
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# KnotNet model
class KnotNet(nn.Module):
    def __init__(self, num_strands=4, hidden_dim=64):
        super().__init__()
        self.num_strands = num_strands
        self.hidden_dim = hidden_dim
        self.initial_state = nn.Parameter(torch.randn(num_strands, hidden_dim))
        self.sub_thetas1 = nn.Parameter(torch.randn(1))
        self.sub_thetas2 = nn.Parameter(torch.randn(2))
        self.meta_thetas = nn.Parameter(torch.randn(3))
        self.gates = nn.Parameter(torch.randn(3))
        self.norm = nn.LayerNorm(hidden_dim)
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
        gate_vals = self.gates.sigmoid()

        for t in range(max_len):
            gen = braids[:, t]
            mask = (gen != 0)
            p = torch.abs(gen) - 1
            s = torch.sign(gen).float()

            sub_mask1 = mask & (p == 0)
            if sub_mask1.any():
                th = self.sub_thetas1[0] * s[sub_mask1] * gate_vals[0]
                cos_th = th.cos().unsqueeze(1)
                sin_th = th.sin().unsqueeze(1)
                u = state[sub_mask1, 0].clone()
                v = state[sub_mask1, 1].clone()
                state[sub_mask1, 0] = u * cos_th - v * sin_th
                state[sub_mask1, 1] = u * sin_th + v * cos_th

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

        for pp in range(3):
            th = self.meta_thetas[pp] * gate_vals[pp]
            cos_th = th.cos()
            sin_th = th.sin()
            u = state[:, pp].clone()
            v = state[:, pp + 1].clone()
            state[:, pp] = u * cos_th - v * sin_th
            state[:, pp + 1] = u * sin_th + v * cos_th

        state = torch.stack([self.norm(state[:, i]) for i in range(self.num_strands)], dim=1)
        flat = state.view(batch_size, -1)
        out = self.mlp(flat)
        class_out = out[:, 0].sigmoid()
        vol_out = out[:, 1]
        return class_out, vol_out

def generate_synthetic_braid(num_strands=4, min_len=5, max_len=50):
    """Generate synthetic braid"""
    gen_range = list(range(1, num_strands)) + list(range(-num_strands+1, 0))
    length = random.randint(min_len, max_len)
    return [random.choice(gen_range) for _ in range(length)]

def load_real_knot_data(max_knots=1000, max_crossing=12):
    """Load real knot data from SnapPy with proper Number handling"""
    print(f"\n{'='*60}")
    print(f"Loading Real Knot Data from SnapPy")
    print(f"{'='*60}")
    
    knot_data = []
    knot_types = defaultdict(list)
    
    # Define knot counts per crossing number
    crossing_limits = {
        3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21, 9: 49, 10: 165, 11: 552, 12: 2176
    }
    
    loaded_count = 0
    failed_count = 0
    
    print("\nLoading knots from standard tables...")
    
    for crossing_num in range(3, min(max_crossing + 1, 13)):
        if crossing_num not in crossing_limits:
            continue
            
        max_index = crossing_limits[crossing_num]
        indices_to_try = min(max_index, max(10, max_knots // 20))
        
        print(f"  Loading {crossing_num}-crossing knots (up to {indices_to_try} of {max_index})...")
        
        for idx in range(1, indices_to_try + 1):
            if loaded_count >= max_knots:
                break
                
            knot_name = f"{crossing_num}_{idx}"
            
            try:
                # Create manifold and link
                M = snappy.Manifold(knot_name)
                link = snappy.Link(knot_name)
                
                # Get volume - handle SnapPy Number objects
                vol = M.volume()
                vol_float = float(vol) if hasattr(vol, '__float__') else vol
                
                is_hyperbolic = 1 if vol_float > 0.1 else 0
                normalized_vol = vol_float / 10.0
                
                # Try to get braid word
                braid = None
                has_real_braid = False
                try:
                    braid_word = link.braid_word()
                    if braid_word:
                        braid = list(braid_word)
                        has_real_braid = True
                except:
                    pass
                
                # If no braid word, generate one
                if not braid:
                    braid_length = random.randint(crossing_num, min(crossing_num * 3, 50))
                    braid = []
                    for _ in range(braid_length):
                        max_gen = min(3, (crossing_num + 1) // 2)
                        gen = random.randint(1, max_gen)
                        if random.random() < 0.5:
                            gen = -gen
                        braid.append(gen)
                
                info = {
                    'name': knot_name,
                    'crossing_number': crossing_num,
                    'volume': vol_float,
                    'braid_length': len(braid),
                    'is_hyperbolic': is_hyperbolic,
                    'has_real_braid': has_real_braid
                }
                
                knot_data.append((braid, is_hyperbolic, normalized_vol, info))
                knot_types[crossing_num].append(len(knot_data) - 1)
                loaded_count += 1
                
                if loaded_count <= 5 or loaded_count % 50 == 0:
                    print(f"    ✓ Loaded {knot_name}: vol={vol_float:.4f}, braid_len={len(braid)}, real_braid={has_real_braid}")
                    
            except Exception as e:
                failed_count += 1
                if failed_count <= 3:
                    print(f"    ✗ Failed {knot_name}: {str(e)[:50]}")
    
    print(f"\nLoaded {loaded_count} knots from standard tables")
    
    # Add from HTLinkExteriors if needed (limited to avoid timeout)
    if loaded_count < max_knots:
        remaining = min(100, max_knots - loaded_count)  # Limit to 100 to avoid timeout
        print(f"\nLoading up to {remaining} additional knots from census...")
        
        try:
            ht_census = snappy.HTLinkExteriors(cusps=1)
            census_count = 0
            
            for M in ht_census:
                if census_count >= remaining:
                    break
                    
                try:
                    vol = M.volume()
                    vol_float = float(vol) if hasattr(vol, '__float__') else vol
                    
                    is_hyperbolic = 1 if vol_float > 0.1 else 0
                    normalized_vol = vol_float / 10.0
                    
                    # Generate braid
                    num_tet = M.num_tetrahedra()
                    braid_length = min(50, max(5, num_tet * 3))
                    braid = []
                    for _ in range(braid_length):
                        gen = random.randint(1, 3)
                        if random.random() < 0.5:
                            gen = -gen
                        braid.append(gen)
                    
                    info = {
                        'name': str(M),
                        'num_tetrahedra': num_tet,
                        'volume': vol_float,
                        'braid_length': len(braid),
                        'is_hyperbolic': is_hyperbolic,
                        'from_census': True
                    }
                    
                    knot_data.append((braid, is_hyperbolic, normalized_vol, info))
                    loaded_count += 1
                    census_count += 1
                    
                    if census_count <= 3 or census_count % 20 == 0:
                        print(f"    Census {census_count}: vol={vol_float:.4f}")
                        
                except Exception as e:
                    continue
            
            print(f"  Loaded {census_count} knots from HTLinkExteriors")
            
        except Exception as e:
            print(f"  Could not access HTLinkExteriors: {str(e)[:100]}")
    
    # Generate synthetic knots if still needed
    if loaded_count < 100:
        synthetic_needed = 100 - loaded_count
        print(f"\nGenerating {synthetic_needed} synthetic knots...")
        
        for i in range(synthetic_needed):
            braid = generate_synthetic_braid()
            
            try:
                link = snappy.Link(braid_closure=braid)
                manifold = link.exterior()
                vol = manifold.volume()
                vol_float = float(vol) if hasattr(vol, '__float__') else vol
                is_hyperbolic = 1 if vol_float > 0.1 else 0
                normalized_vol = vol_float / 10.0
            except:
                # Fallback to heuristics
                complexity = sum(abs(x) for x in braid) / len(braid)
                is_hyperbolic = 1 if complexity > 1.5 else 0
                vol_float = complexity * 2.5 if is_hyperbolic else 0.05
                normalized_vol = vol_float / 10.0
            
            crossing_num = len(braid) // 5
            info = {
                'name': f'synthetic_{i}',
                'crossing_number': crossing_num,
                'volume': vol_float,
                'braid_length': len(braid),
                'synthetic': True
            }
            
            knot_data.append((braid, is_hyperbolic, normalized_vol, info))
            knot_types[crossing_num].append(len(knot_data) - 1)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Data Loading Complete")
    print(f"{'='*60}")
    print(f"Total knots loaded: {len(knot_data)}")
    
    if knot_types:
        print("\nKnots by crossing number:")
        for k in sorted(knot_types.keys())[:10]:
            print(f"  {k} crossings: {len(knot_types[k])} knots")
    
    # Show sample
    if knot_data:
        print("\nSample of loaded knots:")
        for i in range(min(3, len(knot_data))):
            braid, is_hyp, vol, info = knot_data[i]
            print(f"  {info['name']}: hyperbolic={is_hyp}, vol={info['volume']:.4f}, braid_len={len(braid)}")
    
    if len(knot_data) == 0:
        raise ValueError("No knot data could be loaded!")
    
    return knot_data, knot_types

class KnotDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        braid, label, vol, info = self.data[idx]
        return torch.tensor(braid, dtype=torch.long), torch.tensor(label, dtype=torch.float), torch.tensor(vol, dtype=torch.float), info

def collate_fn(batch):
    braids, labels, vols, infos = zip(*batch)
    lengths = [len(b) for b in braids]
    max_len = max(lengths) if lengths else 0
    pad_braids = torch.zeros(len(braids), max_len, dtype=torch.long)
    for i, b in enumerate(braids):
        pad_braids[i, :lengths[i]] = b
    return pad_braids, torch.tensor(labels, dtype=torch.float), torch.tensor(vols, dtype=torch.float), infos

def benchmark_model(train_data, val_data, test_data, device, epochs=30):
    """Train and benchmark the model"""
    
    # Create dataloaders
    batch_size = 32
    train_dataset = KnotDataset(train_data)
    val_dataset = KnotDataset(val_data)
    test_dataset = KnotDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Training
    print(f"\n{'='*60}")
    print("Training Phase")
    print(f"{'='*60}")
    
    model = KnotNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    best_val_acc = 0
    best_model_state = None
    
    train_start = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for braids, labels, vols, _ in train_loader:
            braids, labels, vols = braids.to(device), labels.to(device), vols.to(device)
            
            optimizer.zero_grad()
            class_out, vol_out = model(braids)
            loss = 0.7 * bce_loss(class_out, labels) + 0.3 * mse_loss(vol_out, vols)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            pred = (class_out > 0.5).float()
            train_correct += (pred == labels).sum().item()
            train_total += len(labels)
        
        train_acc = train_correct / train_total
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for braids, labels, vols, _ in val_loader:
                braids, labels, vols = braids.to(device), labels.to(device), vols.to(device)
                class_out, vol_out = model(braids)
                
                pred = (class_out > 0.5).float()
                val_correct += (pred == labels).sum().item()
                val_total += len(labels)
        
        val_acc = val_correct / val_total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
    
    train_time = time.time() - train_start
    print(f"\nTraining completed in {train_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Testing
    print(f"\n{'='*60}")
    print("Testing Phase")
    print(f"{'='*60}")
    
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_correct = 0
    test_total = 0
    test_mse = 0
    inference_times = []
    
    with torch.no_grad():
        for braids, labels, vols, _ in test_loader:
            braids = braids.to(device)
            
            start = time.time()
            class_out, vol_out = model(braids)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - start
            
            inference_times.append(inference_time)
            
            pred = (class_out > 0.5).cpu()
            test_correct += (pred == labels).sum().item()
            test_total += len(labels)
            test_mse += ((vol_out.cpu() - vols) ** 2).sum().item()
    
    test_acc = test_correct / test_total
    test_mse = test_mse / test_total
    throughput = test_total / sum(inference_times)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Volume MSE: {test_mse:.4f}")
    print(f"Throughput: {throughput:.2f} samples/sec")
    
    return {
        'train_time': train_time,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_mse': test_mse,
        'throughput': throughput,
        'model_state': best_model_state
    }

def main():
    """Main benchmarking function"""
    print("="*60)
    print(" KnotNet Benchmarking Suite (Fixed) ".center(60))
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    knot_data, knot_types = load_real_knot_data(max_knots=500, max_crossing=10)
    
    # Split data
    random.shuffle(knot_data)
    n = len(knot_data)
    train_size = int(0.6 * n)
    val_size = int(0.2 * n)
    
    train_data = knot_data[:train_size]
    val_data = knot_data[train_size:train_size + val_size]
    test_data = knot_data[train_size + val_size:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    
    # Run benchmark
    results = benchmark_model(train_data, val_data, test_data, device, epochs=30)
    
    # Save results
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'dataset_size': n,
        'train_time': results['train_time'],
        'best_val_acc': results['best_val_acc'],
        'test_acc': results['test_acc'],
        'test_mse': results['test_mse'],
        'throughput': results['throughput']
    }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    torch.save(results['model_state'], 'best_model.pt')
    
    print(f"\n{'='*60}")
    print(" Benchmark Complete! ".center(60))
    print(f"{'='*60}")
    print("Results saved to: benchmark_results.json")
    print("Model saved to: best_model.pt")

if __name__ == "__main__":
    main()
