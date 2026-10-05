# Fuzzy Bisimulation Supervised Learning Pipeline

This package implements the end-to-end workflow for supervised learning of fuzzy bisimulation similarity on fuzzy knowledge graphs, following Algorithm 1 of *"Learning to scale similarity reasoning in large knowledge graphs"*.

## Directory Structure

```
├── fuzzy_bisimulation/
│   ├── fuzzy_kg.py             # Triple loading, fuzzification, node features
│   ├── exact_bisimulation.py   # Exact fixed-point fuzzy bisimulation solver
│   ├── local_sampling.py       # h-hop local subgraph sampling
│   ├── generate_labels.py      # Dataset generation and ground truth labeling
│   ├── model.py                # GCNConv encoder and pairwise similarity head
│   ├── train.py                # Supervised model training pipeline
│   ├── evaluate.py             # Evaluation, benchmarking, and error analysis
│   └── README.md               # Package documentation
│
├── data/                       # Raw graphs and generated pair datasets
├── checkpoints/                # Saved model weights and training checkpoints
└── results/                    # Metric summaries, logs, and plots
```

## Workflow Overview

1. **Fuzzification (`fuzzy_kg.py`)**:
   Reads `head<TAB>relation<TAB>tail` triples and assigns each one the fuzzy weight
   `w(s, r, s') = Sigmoid(0.6 · Freq(r) + 0.4 · Deg(s, s'))`.

2. **Local Sampling (`local_sampling.py`)**:
   Extracts the subgraph induced by the 2-hop BFS neighborhood of a seed entity, capped at `max_nodes` entities.

3. **Exact Ground Truth (`exact_bisimulation.py`)**:
   Computes the greatest fixed point `B* = gfp(F)` by iterating `B_{k+1} = F(B_k)` from the full relation, where

   ```
   F(B)(s, t) = min_r min_s' [ w(s, r, s') → max_t' ( w(t, r, t') ⊗ B(s', t') ) ]
   ```

   combined with the reverse-direction condition. Gödel (default), Łukasiewicz and product semantics are available.

4. **Dataset Generation (`generate_labels.py`)**:
   Runs the exact solver on every sampled subgraph, labels sampled entity pairs with `y_ij = B*_{G_k}(s_i, s_j)`, and writes `graph.pt`, `train.pt`, `val.pt`, `test.pt` and `meta.json` to `data/<dataset>/processed/`. The 80/10/10 split is made over subgraphs, so pairs of one subgraph never cross partitions.

5. **Training (`train.py`)**:
   Trains `f(s_i, s_j) = Sigmoid(MLP([h_i ‖ h_j ‖ |h_i − h_j|]))` with `h = GCN(G)` using the Huber loss, Adam, and early stopping on validation MAE. Weights saved in `checkpoints/<dataset>/`.

6. **Evaluation (`evaluate.py`)**:
   Reports MAE, MSE, RMSE, Pearson / Spearman / Kendall-Tau correlation and inference time against the exact labels, next to a Common Neighbors baseline, and writes `metrics.json`, `predictions.csv` and `predictions.png` into `results/<dataset>/`.

## Usage

Run the modules from the repository root:

```bash
python -m fuzzy_bisimulation.generate_labels --dataset WN18RR --num_pairs 80000
python -m fuzzy_bisimulation.train --dataset WN18RR
python -m fuzzy_bisimulation.evaluate --dataset WN18RR
```

The paper uses 50,000 / 80,000 / 150,000 labeled pairs for FB15k-237 / WN18RR / YAGO3-10. Every script lists its options with `--help`; the defaults match the paper's configuration (3-layer GCN, hidden dimension 128, 3-layer MLP head, learning rate 1e-3, batch size 128, at most 100 epochs).

Ablation variants (Section 5.6) are trained with `--no_gnn` or `--no_diff`:

```bash
python -m fuzzy_bisimulation.train --dataset WN18RR --no_gnn --checkpoint_dir checkpoints/WN18RR_no_gnn
python -m fuzzy_bisimulation.evaluate --dataset WN18RR --checkpoint_path checkpoints/WN18RR_no_gnn/best_model.pt --results_dir results/WN18RR_no_gnn
```

## Implementation Choices

The paper leaves the following details open; they are fixed here as stated.

- **Fuzzy semantics.** The default is Gödel semantics (minimum t-norm and its residuum). Select another with `--semantics`.
- **Fuzzification scaling.** `Freq(r)` is the triple count of `r` divided by the largest relation count; `Deg(s, s')` is the mean log-scaled degree of the two entities. Both lie in [0, 1].
- **Boundary selection.** When a BFS frontier exceeds the node budget, the entities with the largest total fuzzy weight towards the previous frontier are kept.
- **Initial node features.** Strongest outgoing and incoming fuzzy weight per relation type, plus log-scaled out- and in-degree.
- **GCN propagation.** The encoder uses `torch_geometric`'s `GCNConv` with the fuzzy weights as edge weights, as in the reference `main.py`. Messages follow the edge direction, so each entity aggregates from its predecessors (and itself).
- **Common Neighbors baseline.** The shared-neighbor count is divided by the size of the larger neighborhood to obtain a score in [0, 1].
