# main.py - Three-stage SIGCF pipeline (Phase 1, 2, 3)
# Stages are automatic: LightGCN -> Adapter -> Full SIGCF (3 trials)

import world
import re
import utils.utils as utils
import torch
import torch.nn.functional as F
import numpy as np
from time import time
import Procedure
import dataloader
from os.path import join
import register
from register import dataset
import os
from models.model import LightGCN
from models.SIGCF import LightGCN_Intent

utils.set_seed(world.seed)
print(">>SEED:", world.seed)

CKPT_DIR = world.CKPT_DIR
os.makedirs(CKPT_DIR, exist_ok=True)
RUN_MODE = getattr(world, "RUN_MODE", "tune")
EVAL_FILE = world.config.get("eval_file", "test.txt")
EVAL_NAME = "Val" if os.path.basename(EVAL_FILE).lower().startswith("val") else "Test"

# ============================================================
# Phase 1: Pure LightGCN Pre-training
# ============================================================
print("\n" + "="*70)
print(f"  PHASE 1: LightGCN Pre-training ({world.PHASE1_epochs} epochs)")
print("="*70)

lgn_model = LightGCN(world.config, dataset).to(world.device)
lgn_bpr = utils.BPRLoss(lgn_model, world.config)

lgn_log_file = os.path.join(CKPT_DIR, "lgn_pretrain_log.txt")
lgn_weight_file = os.path.join(CKPT_DIR, "lightgcn_pretrained.pth")

with open(lgn_log_file, "w") as f:
    f.write("epoch,recall5,recall10,recall20,ndcg5,ndcg10,ndcg20\n")

best_recall20 = 0.0
for epoch in range(world.PHASE1_epochs):
    out = Procedure.BPR_train_original(dataset, lgn_model, lgn_bpr, epoch, neg_k=1)
    print(f"[Phase1] EPOCH[{epoch+1}/{world.PHASE1_epochs}] {out}")

    if RUN_MODE == "tune" and epoch % world.EVAL_interval == 0 and epoch >= world.EVAL_interval:
        res = Procedure.Test(dataset, lgn_model, epoch, w=None, multicore=world.config["multicore"])
        r, n = res["recall"], res["ndcg"]
        with open(lgn_log_file, "a") as f:
            f.write(f"{epoch},{r[0]:.6f},{r[1]:.6f},{r[2]:.6f},{n[0]:.6f},{n[1]:.6f},{n[2]:.6f}\n")
        print(f"  {EVAL_NAME} Recall@20={r[2]:.6f}  {EVAL_NAME} NDCG@20={n[2]:.6f}")
        if r[2] > best_recall20:
            best_recall20 = r[2]
            torch.save(lgn_model.state_dict(), lgn_weight_file)
            print(f"  >> [BEST] saved")

if RUN_MODE == "final" and world.PHASE1_epochs > 0:
    torch.save(lgn_model.state_dict(), lgn_weight_file)
    print("  >> [FINAL] saved last LightGCN checkpoint without test/val selection")

print(f"\n[Phase1 Done] Best LightGCN Recall@20={best_recall20:.6f}")

# ============================================================
# Phase 2: Adapter Pre-training
# ============================================================
print("\n" + "="*70)
print(f"  PHASE 2: Adapter Pre-training ({world.ADAPTER_epochs} epochs)")
print("="*70)

del lgn_model, lgn_bpr

adapter_model = LightGCN_Intent(world.config, dataset).to(world.device)

if os.path.exists(lgn_weight_file):
    sd = torch.load(lgn_weight_file, map_location="cpu")
    adapter_model.embedding_user.weight.data.copy_(sd["embedding_user.weight"].to(adapter_model.embedding_user.weight.device))
    adapter_model.embedding_item.weight.data.copy_(sd["embedding_item.weight"].to(adapter_model.embedding_item.weight.device))
    print("[Phase2] LightGCN weights loaded")

adapter_model.init_intent_nodes()

for p in adapter_model.embedding_user.parameters(): p.requires_grad = False
for p in adapter_model.embedding_item.parameters(): p.requires_grad = False
adapter_model.intent_embedding.requires_grad = False

trainable = []
for n, p in adapter_model.named_parameters():
    # Adapter pretrains the remaining semantic projection and its node-adaptive
    # intent-injection gates.  The final node-text semantic branch is still
    # trained only by the joint objective.
    if (
        n.startswith("intent_projection.")
        or n.startswith("mashup_intent_gate.")
        or n.startswith("api_intent_gate.")
    ):
        p.requires_grad = True
        trainable.append(p)
    elif p.requires_grad:
        p.requires_grad = False

adapter_opt = torch.optim.Adam(trainable, lr=0.001)
adapter_log_file = os.path.join(CKPT_DIR, "adapter_pretrain_log.txt")
adapter_weight_file = os.path.join(CKPT_DIR, "adapter_pretrained.pth")

with open(adapter_log_file, "w") as f:
    f.write("epoch,recon_loss\n")

train_users = torch.from_numpy(dataset.trainUser).long().to(world.device)
train_items = torch.from_numpy(dataset.trainItem).long().to(world.device)
bs = 2048

best_loss = float("inf")
for epoch in range(world.ADAPTER_epochs):
    adapter_model.train()
    total, nb = 0.0, 0
    for i in range(0, len(train_users), bs):
        bu, bi = train_users[i:i+bs], train_items[i:i+bs]
        with torch.no_grad():
            orig = torch.sum(adapter_model.embedding_user(bu) * adapter_model.embedding_item(bi), dim=1)
        eu, ei = adapter_model._compute_intent_enhanced(adapter_model.embedding_user.weight, adapter_model.embedding_item.weight)
        enh = torch.sum(eu[bu] * ei[bi], dim=1)
        loss = F.mse_loss(enh, orig)
        adapter_opt.zero_grad(); loss.backward(); adapter_opt.step()
        total += loss.item(); nb += 1
    avg = total / max(nb, 1)
    with open(adapter_log_file, "a") as f:
        f.write(f"{epoch},{avg:.6f}\n")
    print(f"[Phase2] EPOCH[{epoch+1}/{world.ADAPTER_epochs}] ReconLoss={avg:.6f}")
    if avg < best_loss:
        best_loss = avg
        torch.save(adapter_model.state_dict(), adapter_weight_file)
        print(f"  >> [BEST] saved")

print(f"\n[Phase2 Done] Best recon loss={best_loss:.6f}")

# ============================================================
# Phase 3: Full SIGCF Training
# ============================================================
print("\n" + "="*70)
phase3_trial_count = 1 if RUN_MODE == "final" else len(world.TRIAL_seeds)
print(f"  PHASE 3: Full SIGCF Training ({phase3_trial_count} trials x {world.FULL_epochs} epochs)")
print("="*70)

del adapter_model

TRAIN_EPOCHS = world.FULL_epochs
if RUN_MODE == "final":
    trials = [(world.TRIAL_seeds[0], "final")]
    if len(world.TRIAL_seeds) > 1:
        print(f"[Final] use only the first seed {world.TRIAL_seeds[0]} so Test is evaluated once")
else:
    trials = [(seed, f"trial{i + 1}") for i, seed in enumerate(world.TRIAL_seeds)]
all_results = []

for tseed, tname in trials:
    print(f"\n"+ "-"*60)
    print(f"  Trial: {tname}  seed={tseed}")
    print("-"*60)
    utils.set_seed(tseed)

    trial_model = LightGCN_Intent(world.config, dataset).to(world.device)

    if os.path.exists(adapter_weight_file):
        trial_model.load_adapter_pretrained(adapter_weight_file)
        print("  [Load] adapter_pretrained.pth")
    elif os.path.exists(lgn_weight_file):
        trial_model.load_lightgcn_pretrained(lgn_weight_file)
        trial_model.init_intent_nodes()
        print("  [Load] LightGCN weights + fresh intents")

    trial_model.intent_initialized = True
    trial_model.enable_pretrained_mode()

    trial_bpr = utils.BPRLoss(trial_model, world.config)
    wfile = os.path.join(CKPT_DIR, f"full_{tname}.pth")
    logfile = os.path.join(CKPT_DIR, f"full_{tname}_log.txt")

    # Select one checkpoint jointly by Recall@5, then NDCG@5.  The selected
    # checkpoint's complete metric vector is retained for final reporting.
    best_key = (-float("inf"), -float("inf"))
    best = dict.fromkeys(["recall5","recall10","recall20","ndcg5","ndcg10","ndcg20"], 0)
    best_epoch = None

    with open(logfile, "w") as f:
        f.write("epoch,bpr_raw,nca_raw,recall5,recall10,recall20,ndcg5,ndcg10,ndcg20\n")

    for epoch in range(TRAIN_EPOCHS):
        trial_model.current_epoch = epoch
        if hasattr(trial_model, "update_intent_weight"):
            trial_model.update_intent_weight()
        out = Procedure.BPR_train_original(dataset, trial_model, trial_bpr, epoch, neg_k=1)
        print(f"  [{tname}] EPOCH[{epoch+1}/{TRAIN_EPOCHS}] {out}")

        if RUN_MODE == "tune" and epoch % world.EVAL_interval == 0:
            res = Procedure.Test(dataset, trial_model, epoch, w=None, multicore=world.config["multicore"])
            r, n = res["recall"], res["ndcg"]
            bpr_raw = 0.0; nca_raw = 0.0
            m = re.search(r"bpr=([\d.]+)", out)
            if m: bpr_raw = float(m.group(1))
            m = re.search(r"nca_raw=([\d.]+)", out)
            if m: nca_raw = float(m.group(1))
            with open(logfile, "a") as f:
                f.write(f"{epoch},{bpr_raw:.6f},{nca_raw:.6f},"
                        f"{r[0]:.6f},{r[1]:.6f},{r[2]:.6f},{n[0]:.6f},{n[1]:.6f},{n[2]:.6f}\n")
            print(
                f"    {EVAL_NAME} R@20={r[2]:.6f} {EVAL_NAME} N@20={n[2]:.6f} "
                f"BPR={bpr_raw:.4f} NCA={nca_raw:.6f}"
            )

            cur = dict(recall5=r[0], recall10=r[1], recall20=r[2],
                       ndcg5=n[0], ndcg10=n[1], ndcg20=n[2])
            current_key = (cur["recall5"], cur["ndcg5"])
            if current_key > best_key:
                best_key = current_key
                best = cur
                best_epoch = epoch
                torch.save(trial_model.state_dict(), wfile)
                print(
                    f"    >> [BEST @5] epoch={epoch} "
                    f"Recall@5={cur['recall5']:.6f} NDCG@5={cur['ndcg5']:.6f}"
                )

        if RUN_MODE == "tune" and epoch >= 30 and epoch % world.EVAL_interval == 0:
            stall = epoch - best_epoch if best_epoch is not None else 0
            if stall >= world.EARLY_STOP_PATIENCE:
                print(f"  [EarlyStop] at epoch {epoch}")
                break

    if RUN_MODE == "final":
        torch.save(trial_model.state_dict(), wfile)
        res = Procedure.Test(dataset, trial_model, TRAIN_EPOCHS, w=None, multicore=world.config["multicore"])
        r, n = res["recall"], res["ndcg"]
        best = dict(recall5=r[0], recall10=r[1], recall20=r[2],
                    ndcg5=n[0], ndcg10=n[1], ndcg20=n[2])
        with open(logfile, "a") as f:
            f.write(f"{TRAIN_EPOCHS},0.000000,0.000000,0.000000,"
                    f"{r[0]:.6f},{r[1]:.6f},{r[2]:.6f},{n[0]:.6f},{n[1]:.6f},{n[2]:.6f}\n")
        print(f"  [Final Test Once] R@20={best['recall20']:.6f} N@20={best['ndcg20']:.6f}")

    if RUN_MODE == "tune":
        if best_epoch is None or not os.path.exists(wfile):
            raise RuntimeError("No validation checkpoint was selected in Phase 3")
        trial_model.load_state_dict(torch.load(wfile, map_location=world.device))
        test_config = dict(world.config)
        test_config["eval_file"] = world.config.get("test_file", "test.txt")
        test_dataset = dataloader.Loader(config=test_config, path=world.DATA_PATH)
        test_res = Procedure.Test(
            test_dataset, trial_model, best_epoch, w=None,
            multicore=world.config["multicore"],
        )
        test_r, test_n = test_res["recall"], test_res["ndcg"]
        best = dict(recall5=test_r[0], recall10=test_r[1], recall20=test_r[2],
                    ndcg5=test_n[0], ndcg10=test_n[1], ndcg20=test_n[2])
        test_logfile = os.path.join(CKPT_DIR, f"full_{tname}_test_log.txt")
        with open(test_logfile, "w") as f:
            f.write("best_val_epoch,recall5,recall10,recall20,ndcg5,ndcg10,ndcg20\n")
            f.write(f"{best_epoch},{test_r[0]:.6f},{test_r[1]:.6f},{test_r[2]:.6f},"
                    f"{test_n[0]:.6f},{test_n[1]:.6f},{test_n[2]:.6f}\n")
        print(f"  [Final Test @ best val epoch {best_epoch}] "
              f"R@5={best['recall5']:.6f} N@5={best['ndcg5']:.6f} "
              f"R@20={best['recall20']:.6f} N@20={best['ndcg20']:.6f}")

    all_results.append(best)
    print(f"  {tname} result: R@20={best['recall20']:.6f} N@20={best['ndcg20']:.6f}")

print(f"\n"+ "="*70)
title = "FINAL TEST RESULTS (mean +/- std)"
print(title)
print("="*70)
for k in ["recall5","recall10","recall20","ndcg5","ndcg10","ndcg20"]:
    v = np.array([r[k] for r in all_results])
    print(f"  {k}: {v.mean():.6f} +/- {v.std():.6f}")
print("="*70)
