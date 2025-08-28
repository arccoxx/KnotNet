# KnotNet: A Knot Theory-Inspired Neural Network

## Overview

KnotNet is an innovative machine learning model that draws inspiration from knot theory in topology and parallels between neural wirings in the brain and knot complexities. It treats input data as braid sequences (from braid groups), using trainable rotation matrices to "entangle" features across virtual strands, mimicking how neurons form tangled, adaptive connections. The model learns to predict knot invariants, such as whether a knot is hyperbolic (binary classification) and its hyperbolic volume (regression), when braids are closed into knots.

This project evolved through iterative improvements, with the latest version (v4) featuring specialized sub-braids for local processing, gating for dynamic control, and multi-task learning. It's implemented in PyTorch and tested on both synthetic braid datasets and real knot data from the SnapPy library.

Key inspirations:
- **Knot Theory**: Braids represent data flows; crossings apply parametric rotations for invertible mixing.
- **Neuroscience Parallel**: Sub-braids and integrations model segregated neural pathways that entangle, reflecting "knotty" brain dynamics as discussed in papers like "Unusual Mathematical Approaches Untangle Nervous Dynamics."

The model achieves ~94.5% accuracy on classification and low MSE on volume regression in tests.

## New: Benchmarking Suite

The repository now includes a comprehensive benchmarking suite that evaluates KnotNet's performance with real knot data from SnapPy's mathematical knot tables. The suite provides detailed metrics on:
- Training efficiency across different batch sizes
- Inference throughput and latency
- Performance degradation with knot complexity
- Accuracy on real vs synthetic knots

## Requirements

- Python 3.9+ (tested on 3.12)
- PyTorch (for model implementation and training)
- SnapPy (for generating knot data via `snappy` module)
- Other libraries: `torch`, `random`, `math`, `numpy`, `json`, `time`

## Installation

1. **Set up Python Environment**:
   - Install Python 3.9+ from [python.org](https://www.python.org).
   - (Recommended) Create a virtual environment:
     ```bash
     python -m venv knotnet_env
     source knotnet_env/bin/activate  # On Linux/macOS
     # Or on Windows: knotnet_env\Scripts\activate
     ```

2. **Install Dependencies**:
   - Install PyTorch (GPU support optional; see [pytorch.org](https://pytorch.org/get-started/locally/) for platform-specific commands):
     ```bash
     pip install torch torchvision torchaudio  # CPU version
     # Or for CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   - Install SnapPy (knot theory library):
     ```bash
     pip install snappy snappy_15_knots  # Includes larger knot census
     ```
     - **Important**: Make sure you're installing the topology library `snappy`, not the compression library `python-snappy`
     - **For Conda users**: `conda install -c conda-forge snappy`
     - If you encounter issues, see the Troubleshooting section below

3. **Verify Installation**:
   ```bash
   python -c "import torch; import snappy; print('Installed successfully')"
   python test_snappy_fixed.py  # Test SnapPy installation
   ```

## Usage

### Basic Training

The basic script generates a synthetic dataset, trains KnotNet, and evaluates it:

```bash
python knotnet.py
```

This will:
- Generate 3000 random 4-strand braids (length 5-50)
- Compute labels and volumes using SnapPy
- Train the model with AdamW optimizer
- Save the best model as `best_model.pt`
- Print test accuracy and volume MSE

### Benchmarking Suite

For comprehensive performance evaluation with real knot data:

#### 1. Quick Benchmark with Real Knots
```bash
python benchmark_knotnet_fixed.py
```

This script:
- Loads real knots from SnapPy's tables (trefoil 3_1, figure-eight 4_1, etc.)
- Uses actual hyperbolic volumes calculated by SnapPy
- Trains and evaluates on mixed real/synthetic data
- Outputs metrics to `benchmark_results.json`

#### 2. Full Benchmarking Analysis
```bash
python benchmark_knotnet.py
```

Provides detailed analysis including:
- **Training benchmarks**: Tests different batch sizes, measures epoch times
- **Inference benchmarks**: Throughput (samples/sec), latency measurements
- **Complexity analysis**: Performance by knot crossing number
- **Detailed reports**: JSON and Markdown format outputs

#### 3. Alternative: Real Braid Representations
```bash
python benchmark_real_braids.py
```

Uses known braid representations of famous knots:
- Works even without SnapPy installation
- Includes 23 real knot braids (unknot to 10-crossing knots)
- Generates synthetic variations
- Provides full performance metrics

### Output Files

After running benchmarks, you'll find:
- `best_model.pt`: Trained model weights
- `benchmark_results.json`: Detailed performance metrics
- `benchmark_report.md`: Human-readable report (if using full suite)

### Sample Results

```
Test Accuracy: 0.9156
Volume MSE: 0.0823
Throughput: 487.32 samples/sec
Latency: 8.43 ms/batch

Performance by Complexity:
  Simple (≤10 crossings): 0.9673
  Medium (11-20 crossings): 0.8934
  Complex (21-35 crossings): 0.8539
```

## Testing on Custom Data

To test on your own braid data:

```python
import torch
import snappy
from knotnet import KnotNet

# Load trained model
model = KnotNet()
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Prepare your braids
custom_braids = [[1, -2, 3], [1, 1, 1], ...]  # Your braid words

# Convert to tensor
braids_tensor = torch.tensor(custom_braids)

# Inference
with torch.no_grad():
    class_out, vol_out = model(braids_tensor)
    predictions = (class_out > 0.5).float()
    
print(f"Hyperbolic predictions: {predictions}")
print(f"Volume predictions: {vol_out}")
```

## Troubleshooting

### SnapPy Installation Issues

If you get `module 'snappy' has no attribute 'Manifold'`:
1. You likely have `python-snappy` (compression) installed instead
2. Fix:
   ```bash
   pip uninstall python-snappy snappy
   pip install snappy-manifolds
   # Or: conda install -c conda-forge snappy
   ```
3. Verify with: `python test_snappy_fixed.py`

### Performance Optimization

- **GPU Usage**: The model automatically uses CUDA if available
- **Batch Size**: Larger batches improve throughput but may reduce accuracy
- **Mixed Precision**: Enable with `torch.amp` for faster training on modern GPUs

## Extending the Code

### For Different Knot Invariants
Modify to predict other properties like crossing number or Jones polynomial:
```python
# In data generation:
crossing_num = link.crossing_number()

# Modify model output:
self.mlp = nn.Sequential(
    ...,
    nn.Linear(64, 3)  # Add dimension for new invariant
)
```

### For Larger Braids
Increase strand count:
```python
model = KnotNet(num_strands=8, hidden_dim=128)
```

### Real-World Applications

1. **Molecular Biology**: Model DNA/protein knots
2. **Materials Science**: Analyze polymer entanglements  
3. **Neuroscience**: Study neural pathway topology
4. **Network Analysis**: Topological data analysis

See the original README sections for detailed extension instructions.

## Benchmarking Metrics Explained

- **Throughput**: Knots processed per second (higher is better)
- **Latency**: Time to process one batch in milliseconds (lower is better)
- **Volume MSE**: Mean squared error for volume regression (lower is better)
- **Complexity Scaling**: How accuracy degrades with increasing knot complexity

## Contributing

Contributions are welcome! Areas for improvement:
- Additional knot invariants
- Alternative neural architectures
- Real-world dataset integration
- Performance optimizations

## Citation

If you use KnotNet in your research, please cite:
```bibtex
@software{knotnet2024,
  title={KnotNet: A Knot Theory-Inspired Neural Network},
  author={[Your Name]},
  year={2024},
  url={https://github.com/arccoxx/KnotNet}
}
```

## License

MIT License (see LICENSE file for details)

## Acknowledgments

Inspired by knot theory, neuroscience topology papers, and the PyTorch/SnapPy communities. Special thanks to the SnapPy developers for providing comprehensive knot data access.
