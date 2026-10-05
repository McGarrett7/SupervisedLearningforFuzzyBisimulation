# SupervisedLearningforFuzzyBisimulation

Supervised learning framework that approximates fuzzy bisimulation similarity on large fuzzy knowledge graphs. Exact fuzzy bisimulation values, computed by fixed-point iteration on sampled local subgraphs, supervise a GNN-based pairwise predictor, so that later similarity queries need a single forward pass instead of a new fixed-point computation.

The implementation follows *"Learning to scale similarity reasoning in large knowledge graphs"* (T. H. K. Nguyen, T. H. Tran) and builds on the reference repository [AltacomLab/SupervisedLearningforFuzzyBisimulation](https://github.com/AltacomLab/SupervisedLearningforFuzzyBisimulation).

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download `FB15k-237.rar`, `WN18RR.rar` and `YAGO3-10.rar` from the reference repository and extract them into `data/`, so that each dataset has its own folder:

```
data/
├── FB15k-237/train.txt
├── WN18RR/train.txt
└── YAGO3-10/train.txt
```

Each `train.txt` holds one `head<TAB>relation<TAB>tail` triple per line. The optional `entities.dict` / `relations.dict` files fix the id order; when one is missing or malformed, the ids are rebuilt from the triples.

## Quick Start

```bash
python -m fuzzy_bisimulation.generate_labels --dataset WN18RR --num_pairs 80000
python -m fuzzy_bisimulation.train --dataset WN18RR
python -m fuzzy_bisimulation.evaluate --dataset WN18RR
```

Generated datasets are written to `data/<dataset>/processed/`, model weights to `checkpoints/<dataset>/`, and metrics and plots to `results/<dataset>/`.

See [fuzzy_bisimulation/README.md](fuzzy_bisimulation/README.md) for the module-by-module description and the implementation choices.
