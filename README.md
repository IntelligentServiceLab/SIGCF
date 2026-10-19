# SIGCF: Semantic Intent-Aware Graph Collaborative Filtering

SIGCF is an innovative semantic intent-aware graph collaborative filtering framework designed to overcome the critical interaction sparsity challenge
in Web API recommendation. The model combines:

- LightGCN collaborative message propagation;
- semantic intent nodes and intent-aware propagation;
- NCA representation learning;
- separate Mashup/API gates for fusing intent and node-text semantics.

Training consists of LightGCN pre-training, adapter pre-training, and joint
SIGCF training.

## Requirements

```bash
pip install -r requirements.txt
```

The implementation uses PyTorch 2.1.0, DGL, and Sentence Transformers. If a
local SBERT model is required, pass its path with `--sbert_model_name`.

## Main experiment

Run the following command from the directory containing `main.py`:

```bash
python main.py \
  --model lgn_intent \
  --run_mode tune \
  --trial_seeds 2020,2021,2022 \
  --ckpt_dir runs/sigcf_main
```

In `tune` mode, the model trains the three stages, selects the best checkpoint
according to Recall@5 and NDCG@5 on the configured evaluation split, and then
reports test-set results for each seed. Training logs, checkpoints, and test
results are saved under `--ckpt_dir`.

After selecting the final hyperparameters, a single final training run can be
performed with:

```bash
python main.py \
  --model lgn_intent \
  --dataset data \
  --run_mode final \
  --trial_seeds 2020 \
  --ckpt_dir runs/sigcf_final
```

The main default settings are defined in `parse.py`, including embedding
dimension 320, 150 full-training epochs, early-stopping patience 20,
`lambda_nca=0.01`, and three trial seeds (`2020,2021,2022`).
