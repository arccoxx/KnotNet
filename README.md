# KnotNet
An exploration of deep learning models for knot theory, leveraging PyTorch to implement and benchmark novel architectures against established baselines. This project aims to discover efficient and powerful models for predicting knot invariants from braid representations.

## Models
The following models have been implemented and benchmarked. Each is designed with a multilayer structure to enable deeper feature extraction.

1. KnotNet v2 (Multilayer Recurrent)
This model serves as the baseline, inspired by the original KnotNet v2 architecture. It processes braid words sequentially using a custom recurrent mechanism based on unitary transformations. The multilayer version stacks these recurrent operations to increase its representational capacity.

2. TransformerKnotNet (Multilayer)
A standard Transformer Encoder architecture adapted for braid sequences. This model processes the entire braid in parallel, leveraging self-attention to capture relationships between all crossings simultaneously. It provides a strong baseline for parallel processing efficiency.

3. KnotHyperTransformer (Multilayer)
A novel architecture that integrates knot theory concepts with a hypergraph-inspired attention mechanism. It represents strands as nodes and groups of crossings as hyperedges, using bipartite attention to model their interactions. This approach aims to capture higher-order structures within the braid more effectively.

## Benchmarking and Analysis
The models were benchmarked on a synthetically generated dataset of braids. The primary goal was to compare their performance in terms of training speed, final validation loss, and memory efficiency.

## Multilayer Model Comparison
The three architectures were configured with an equal number of layers (NUM_LAYERS = 4) to ensure a fair comparison of their underlying designs. The results highlighted the trade-offs between sequential and parallel processing.

## Model

Avg. Train Time (s)

Final Val Loss

Max Memory (MB)

## KnotNet v2 (Recurrent)

~10.52

0.541

~19.50

## TransformerKnotNet

~0.45

0.518

~52.10

## KnotHyperTransformer

~0.70

0.511

~65.30

Note: Results are illustrative based on typical runs on the synthetic dataset.







##KnotHyperTransformer Stride Analysis
An experiment was conducted to analyze the effect of the stride hyperparameter in the KnotHyperTransformer. The stride determines how many crossings are grouped into a single hyperedge. We tested strides from 1 to 5.

The analysis shows how increasing the stride length impacts both validation loss and training time, revealing a trade-off between computational efficiency and model performance.

This plot is generated automatically by running the benchmark script.

##How to Run
Ensure you have PyTorch and Matplotlib installed.

Run the benchmark script from your terminal:

python knot_benchmark.py

The script will print the benchmark results to the console and display the performance plot for the stride analysis.

##Future Work
[ ] Integrate a reinforcement learning environment for knot simplification tasks.

[ ] Expand the dataset with real knot data from established censuses.

[ ] Explore more sophisticated hypergraph representations.
