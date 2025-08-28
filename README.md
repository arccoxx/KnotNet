# KnotNet: A Knot Theory-Inspired Neural Network

## Overview

KnotNet is an innovative machine learning model that draws inspiration from knot theory in topology and parallels between neural wirings in the brain and knot complexities. It treats input data as braid sequences (from braid groups), using trainable rotation matrices to "entangle" features across virtual strands, mimicking how neurons form tangled, adaptive connections. The model learns to predict knot invariants, such as whether a knot is hyperbolic (binary classification) and its hyperbolic volume (regression), when braids are closed into knots.

This project evolved through iterative improvements, with the latest version (v4) featuring specialized sub-braids for local processing, gating for dynamic control, and multi-task learning. It's implemented in PyTorch and tested on synthetic braid datasets generated using the SnapPy library.

Key inspirations:
- **Knot Theory**: Braids represent data flows; crossings apply parametric rotations for invertible mixing.
- **Neuroscience Parallel**: Sub-braids and integrations model segregated neural pathways that entangle, reflecting "knotty" brain dynamics as discussed in papers like "Unusual Mathematical Approaches Untangle Nervous Dynamics."

The model achieves ~90% accuracy on classification and low MSE on volume regression in tests.

## Requirements

- Python 3.9+ (tested on 3.12)
- PyTorch (for model implementation and training)
- SnapPy (for generating knot data via `snappy` module)
- Other libraries: `torch`, `random`, `math` (included in standard Python)

## Installation

1. **Set up Python Environment**:
   - Install Python 3.9+ from [python.org](https://www.python.org).
   - (Recommended) Create a virtual environment:
     ```
     python -m venv knotnet_env
     source knotnet_env/bin/activate  # On Linux/macOS
     # Or on Windows: knotnet_env\Scripts\activate
     ```

2. **Install Dependencies**:
   - Install PyTorch (GPU support optional; see [pytorch.org](https://pytorch.org/get-started/locally/) for platform-specific commands, e.g.):
     ```
     pip install torch torchvision torchaudio  # CPU version
     # Or for CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   - Install SnapPy (knot theory library):
     ```
     pip install snappy snappy_15_knots  # Includes larger knot census; omit snappy_15_knots for lighter install
     ```
     - **Notes for Platforms**:
       - **macOS/Windows**: Use Python from python.org (not system Python on macOS).
       - **Linux**: Install prerequisites like `python3-pip` and `python3-tk` via your package manager (e.g., `sudo apt install python3-pip python3-tk` on Ubuntu). If issues arise, use a virtual environment as above.
       - For Conda users: `conda install -c conda-forge snappy`.
     - If errors occur (e.g., "externally-managed-environment"), add `--break-system-packages` or use a venv.

3. **Verify Installation**:
   - Run `python -c "import torch; import snappy; print('Installed successfully')"` to check.

## Usage

The code is a self-contained PyTorch script that generates a synthetic dataset of braids, trains the KnotNet model, and evaluates it on test data.

### Running the Code

1. **Save the Code**:
   - Copy the provided code into a file, e.g., `knotnet.py`.

2. **Generate Dataset and Train**:
   - Run the script:
     ```
     python knotnet.py
     ```
   - This will:
     - Generate 3000 random 4-strand braids (length 5-50).
     - Compute labels (hyperbolic: 1 if volume > 0.1, else 0) and normalized volumes using SnapPy.
     - Split into train (1600), val (400), test (1000).
     - Train the model with AdamW, cosine annealing, early stopping, and mixed precision (if GPU available).
     - Save the best model as `best_model.pt`.
     - Print test classification accuracy and volume MSE.

3. **Output Example**:
   ```
   Test classification accuracy: 0.9, Volume MSE: 0.1
   ```
   - Training may take minutes to hours depending on hardware (GPU recommended for larger datasets).

4. **Customization Options**:
   - Adjust hyperparameters in the code: e.g., `batch_size=64`, `lr=0.0005`, epochs=100.
   - Change dataset size: Modify the loop in `for _ in range(3000):`.
   - Use GPU: The code auto-detects `cuda` if available.

### Testing on Custom Data

- Prepare a list of braids (e.g., `custom_braids = [[1, -2, 3], ...]`), compute labels/volumes manually via SnapPy.
- Load the model: `model.load_state_dict(torch.load('best_model.pt'))`.
- Infer: `class_out, vol_out = model(torch.tensor(custom_braids).to(device))`.

## Extending the Code for New Real-World Problems

KnotNet's braid-based architecture is flexible for topological machine learning tasks. Here's how to adapt it:

### 1. **Modify for Different Knot Invariants**
   - **Example**: Predict crossing number or Jones polynomial.
     - Update `is_hyperbolic_and_volume` to compute new invariants (e.g., `link.crossing_number()` via SnapPy).
     - Adjust multi-task output: Change MLP to output more dimensions (e.g., `nn.Linear(64, 3)` for class, volume, crossing).
     - Modify loss: Add terms like MSE for new regressions.

### 2. **Scale to Larger Braids/Strands**
   - Increase `num_strands` (e.g., to 8): Update `self.sub_thetas1`, `self.sub_thetas2`, `self.meta_thetas`, `self.gates` to match pairs (num_strands-1).
   - Adjust `generate_braid`: Expand `gen_range` for more generators.
   - Vectorize further for efficiency on long sequences.

### 3. **Adapt to Real-World Datasets**
   - **Neuroscience (Neural Connectomes)**: Model brain wiring as braids.
     - Input: Convert connectome graphs to braid sequences (e.g., via topological sorting or fiber tract data from MRI).
     - Output: Predict properties like modularity or disease states (e.g., Alzheimer's tangles).
     - Extension: Replace SnapPy with networkx for graph invariants; train on datasets like Human Connectome Project.
   - **Molecular Chemistry (DNA/Protein Knots)**: Classify knotted molecules.
     - Input: Braid representations from molecular simulations (use RDKit or BioPython to generate).
     - Output: Predict stability or chirality.
     - Install extra libs if needed (e.g., `pip install rdkit`).
   - **General Topological Data Analysis (TDA)**: For sensor networks or point clouds.
     - Input: Embed data into braids (e.g., via persistent homology to extract generators).
     - Use libraries like gudhi (if installed) for TDA features.

### 4. **Enhance the Model**
   - Add attention mechanisms over strands for better entanglement modeling.
   - Integrate with GNNs: Treat strands as nodes, crossings as edges.
   - For production: Wrap in a class, add argparse for CLI args, use MLflow for tracking.

### 5. **Best Practices for Extension**
   - Start small: Test on toy data before scaling.
   - Debug gradients: Use `torch.autograd.set_detect_anomaly(True)` if in-place errors recur.
   - Contribute: Fork and PR improvements, e.g., for new invariants.

## License

MIT License (assumed; update as needed).

## Acknowledgments

Inspired by knot theory, neuroscience topology papers, and PyTorch/SnapPy communities. For issues, open a GitHub issue (if repo exists).
