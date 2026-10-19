# SIGCF: Semantic Intent Graph Collaborative Filtering

SIGCF is a LightGCN-based Mashup/API recommendation model with a semantic
intent graph. The implementation has three stages:

1. LightGCN pre-training on the interaction graph;
2. adapter pre-training for semantic-intent propagation;
3. joint SIGCF training with semantic-intent propagation, NCA, and the
   node-level SBERT semantic fusion gates.

The model uses separate Mashup and API semantic gates. The semantic-intent
representation is updated before NCA and is then fused with node-text
features by a learned gate.

## Requirements

Python 3.9+ is recommended. Install the dependencies listed in
`requirements.txt`:

```bash
pip install -r requirements.txt
```

The project currently expects PyTorch 2.1.0. Install the DGL wheel matching
your PyTorch/CUDA installation; for the CUDA 12.1 environment used during
development:

```bash
pip install torch==2.1.0
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
```

The model uses `sentence-transformers` to encode Mashup/API text.
`--sbert_model_name` accepts either a Hugging Face model name or a local model
directory. The default is `all-MiniLM-L6-v2`.

## Project and data layout

Run commands from this directory (`SIGCF/SIGCF`). By default, the code looks
for data in the sibling directory `../data`:

```text
SIGCF/
├── SIGCF/
│   ├── main.py
│   ├── parse.py
│   ├── world.py
│   ├── dataloader.py
│   ├── Procedure.py
│   ├── models/
│   └── utils/
└── data/
    ├── train.txt
    ├── test.txt
    ├── train_inner.txt       # optional train/validation protocol
    ├── val.txt               # optional validation split
    ├── Mashup_desc.csv
    ├── API_desc.csv
    ├── mashups.xlsx
    └── apisData.xlsx
```

Each interaction file is a whitespace-separated list of Mashup/API IDs. The
catalog files must use IDs aligned with the interaction graph. A different
data directory can be supplied with `--data_path`.

## Recommended train/validation/test protocol

For hyperparameter selection or early stopping, use `train_inner.txt` for
training, `val.txt` for validation, and keep `test.txt` untouched until final
evaluation:

```bash
python main.py \
  --model lgn_intent \
  --dataset pw \
  --data_path ../data \
  --train_file train_inner.txt \
  --eval_file val.txt \
  --test_file test.txt \
  --run_mode tune \
  --trial_seeds 2020,2021,2022 \
  --ckpt_dir runs/sigcf_tune
```

In `tune` mode, the best checkpoint is selected by Recall@5 and then NDCG@5
on `--eval_file`. After selection, the code evaluates that checkpoint on
`--test_file` and writes one test log for each trial seed.

After choosing all hyperparameters, retrain on the complete training set and
evaluate once on the test set:

```bash
python main.py \
  --model lgn_intent \
  --dataset pw \
  --data_path ../data \
  --train_file train.txt \
  --test_file test.txt \
  --run_mode final \
  --trial_seeds 2020 \
  --ckpt_dir runs/sigcf_final
```

`final` mode does not use validation-based early stopping; it trains for the
configured number of full-training epochs and evaluates once at the end.

## Default parameters

The defaults are defined in `parse.py`:

| Argument | Default |
|---|---:|
| `recdim` | 320 |
| `layer` | 1 |
| `bpr_batch` | 1024 |
| `lr` | 0.002 |
| `decay` | 0.001 |
| `phase1_epochs` | 100 |
| `adapter_epochs` | 10 |
| `full_epochs` | 150 |
| `eval_interval` | 10 |
| `early_stop_patience` | 20 |
| `intent_weight_fixed` | 1.5 |
| `lambda_nca` | 0.01 |
| `nca_temperature` | 0.10 |
| `nca_weight_ui` | 2.0 |
| `nca_weight_ii` | 1.0 |
| `nca_delay` | 5 |
| `topm_mashup` | 19 |
| `warmup_epochs` | 50 |
| `semantic_lr` | 0.00005 |
| `semantic_gate_init_bias` | 0.0 |
| `trial_seeds` | `2020,2021,2022` |

All command-line arguments in `parse.py` can be overridden without changing
the source code. Checkpoint files, per-epoch logs, and test logs are written
under the directory passed to `--ckpt_dir`.

## Outputs

Typical output files include:

```text
lgn_pretrain_log.txt
adapter_pretrain_log.txt
full_trial1_log.txt
full_trial1_test_log.txt
full_trial1.pth
```

The `full_trial*_log.txt` files contain per-epoch training and evaluation
metrics. In `tune` mode, `full_trial*_test_log.txt` records test metrics at
the checkpoint selected on the evaluation split.

## Reproducibility

Set `--seed` for preprocessing and `--trial_seeds` for repeated full-training
trials. For a three-seed experiment, use:

```bash
python main.py --data_path ../data \
  --train_file train_inner.txt --eval_file val.txt --test_file test.txt \
  --run_mode tune --trial_seeds 2020,2021,2022 \
  --ckpt_dir runs/sigcf_3seeds
```

Use the same data split, model checkpoint directory, software versions, and
SBERT model for comparable results.
