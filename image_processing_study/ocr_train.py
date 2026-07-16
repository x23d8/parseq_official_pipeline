"""Train one CRNN OCR model for one image processing method.

Mirrors the structure of
``train_no_refinement/parseq_official_anpr_pipeline.py`` (same style of
config dataclass / fit loop / checkpoint format) so results from both
pipelines are easy to read side by side, but this module trains the small
from-scratch :class:`~image_processing_study.model.CRNN` instead of PARSeq.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from image_processing_study.common import edit_distance, normalize_plate_text
from image_processing_study.dataset import PlateOCRDataset, build_split, collate_batch
from image_processing_study.methods import Method
from image_processing_study.model import BLANK_IDX, CRNN, ctc_greedy_decode, encode_targets

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "image_processing_study" / "experiment_a"
SPLIT_SEED = 42  # shared by every method -- do not vary per run, see dataset.build_split


@dataclass
class OCRTrainConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    epochs: int = 100
    patience: int = 10
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    seed: int = 42
    split_seed: int = SPLIT_SEED
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    limit_train: int | None = None
    limit_val: int | None = None
    limit_test: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(method: Method, cfg: OCRTrainConfig):
    split = build_split(seed=cfg.split_seed, val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio)
    train_samples, val_samples, test_samples = split["train"], split["val"], split["test"]
    if cfg.limit_train is not None:
        train_samples = train_samples[: cfg.limit_train]
    if cfg.limit_val is not None:
        val_samples = val_samples[: cfg.limit_val]
    if cfg.limit_test is not None:
        test_samples = test_samples[: cfg.limit_test]

    loader_kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate_batch)
    train_loader = DataLoader(PlateOCRDataset(train_samples, method), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(PlateOCRDataset(val_samples, method), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(PlateOCRDataset(test_samples, method), shuffle=False, **loader_kwargs)
    sizes = {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)}
    return train_loader, val_loader, test_loader, sizes


def train_one_epoch(model: CRNN, loader: DataLoader, optimizer, loss_fn, cfg: OCRTrainConfig, device: torch.device, epoch: int) -> dict:
    model.train()
    totals = {"loss": 0.0, "samples": 0}
    for images, labels, _paths in tqdm(loader, desc=f"train epoch {epoch}", leave=False):
        images = images.to(device, non_blocking=True)
        targets, target_lengths = encode_targets(labels)
        optimizer.zero_grad(set_to_none=True)
        log_probs = model(images)
        input_lengths = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long)
        loss = loss_fn(log_probs, targets.to(device), input_lengths, target_lengths)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        batch_size = images.shape[0]
        totals["loss"] += float(loss.detach().item()) * batch_size
        totals["samples"] += batch_size
    return {"train_loss": totals["loss"] / max(totals["samples"], 1), "train_samples": totals["samples"]}


@torch.no_grad()
def evaluate(model: CRNN, loader: DataLoader, device: torch.device, split_name: str = "val") -> tuple[dict, pd.DataFrame]:
    model.eval()
    rows = []
    exact = 0
    edits = 0
    chars = 0
    total = 0
    for images, labels, paths in tqdm(loader, desc=f"eval {split_name}", leave=False):
        images = images.to(device, non_blocking=True)
        log_probs = model(images)
        preds, confs = ctc_greedy_decode(log_probs)
        for path, pred, target, conf in zip(paths, preds, labels, confs):
            pred = normalize_plate_text(pred)
            target = normalize_plate_text(target)
            dist = edit_distance(pred, target)
            ok = pred == target
            exact += int(ok)
            edits += dist
            chars += max(len(target), 1)
            total += 1
            rows.append({"image_path": path, "target": target, "prediction": pred, "exact": ok, "edit_distance": dist, "confidence": conf})
    metrics = {
        "split": split_name,
        "samples": total,
        "exact_acc": exact / max(total, 1),
        "cer": edits / max(chars, 1),
        "char_acc": 1.0 - edits / max(chars, 1),
    }
    return metrics, pd.DataFrame(rows)


def save_checkpoint(path: Path, model: CRNN, method_name: str, cfg: OCRTrainConfig, epoch: int, metrics: dict) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "method_name": method_name,
            "config": asdict(cfg),
            "epoch": int(epoch),
            "metrics": metrics,
            "architecture": "image_processing_study_crnn",
        },
        path,
    )


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[CRNN, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CRNN()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, ckpt


def fit(method: Method, cfg: OCRTrainConfig, device: str | torch.device | None = None) -> dict:
    set_seed(cfg.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(cfg.output_dir) / method.name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, sizes = make_loaders(method, cfg)
    model = CRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    history = []
    best_val_exact = -1.0
    epochs_without_improvement = 0
    early_stopped = False
    best_path = output_dir / "best_model.pt"
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, cfg, device, epoch)
        val_metrics, _ = evaluate(model, val_loader, device, split_name="val")
        row = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}, "seconds": time.time() - start}
        history.append(row)
        if val_metrics["exact_acc"] > best_val_exact:
            best_val_exact = val_metrics["exact_acc"]
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, method.name, cfg, epoch, val_metrics)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                early_stopped = True
                print(f"[{method.name}] early stopping at epoch {epoch} "
                      f"(no val_exact_acc improvement for {cfg.patience} epochs, best={best_val_exact:.4f})")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)

    best_model, best_ckpt = load_checkpoint(best_path, device=device)
    test_metrics, test_rows = evaluate(best_model, test_loader, device, split_name="test")
    test_rows.to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "method": method.name,
        "chapter": method.chapter,
        "description": method.description,
        "config": asdict(cfg),
        "best_val_exact": best_val_exact,
        "best_epoch": best_ckpt.get("epoch"),
        "epochs_run": len(history),
        "early_stopped": early_stopped,
        "test_metrics": test_metrics,
        "dataset_sizes": sizes,
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
