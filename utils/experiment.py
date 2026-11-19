"""Shared protocol, metrics and provenance; importing this module runs no jobs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from config import Config
from data.dataset import (
    LABELS, EAVFeatureDataset, active_mask_for_run, feature_paths,
    make_loader, manifest_identity_sha256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORMAL_SETTINGS = {
    "model": "residual_align_fmpa_v5", "run": "eav", "seed": 42,
    "epochs": 60, "patience": 10, "batch_size": 16, "eval_batch_size": 64,
    "lr": 1e-4, "weight_decay": 1e-4, "dropout": 0.2, "grad_clip": 1.0,
    "token_dim": 256, "hidden_dim": 512, "num_heads": 8, "num_classes": 5,
    "temporal_slots": 16, "v5_local_radius": 1, "v5_residual_gate_init": 0.1,
    "v5_alignment_aux_weight": 0.1, "v5_fmpa_aux_weight": 0.1,
    "eeg_feature_name": "EEG-TDE-16", "audio_feature_name": "audio-frame-lld",
    "video_feature_name": "EVA-ViT-G-5s",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    # Preserve the original formal runner's CPU fallback and CUDA AMP policy.
    return torch.device(requested if torch.cuda.is_available() else "cpu")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_provenance():
    paths = [REPOSITORY_ROOT / name for name in ("config.py", "train.py", "evaluate.py", "requirements.txt")]
    for directory in ("model", "data", "utils"):
        paths.extend((REPOSITORY_ROOT / directory).rglob("*.py"))
    files = {str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): sha256_file(path)
             for path in sorted(set(paths)) if path.is_file()}
    return {
        "files_sha256": files,
        "combined_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
    }


def split_provenance(manifest, split_rows):
    return {
        "manifest_sha256": sha256_file(manifest),
        "splits": {split: {
            "count": len(rows), "identity_sha256": manifest_identity_sha256(rows),
            "sample_ids_sha256": hashlib.sha256(json.dumps(
                [row["sample_id"] for row in rows], separators=(",", ":")
            ).encode()).hexdigest(),
        } for split, rows in split_rows.items()},
    }


def validate_formal_split_protocol(config, split_rows):
    if config.smoke:
        return
    path = REPOSITORY_ROOT / "splits" / "eav_manifest_subject_wise_4x5s_60_10_30_seed42.provenance.json"
    with path.open(encoding="utf-8") as handle:
        frozen = json.load(handle)
    actual_hashes = {split: manifest_identity_sha256(rows) for split, rows in split_rows.items()}
    actual_sizes = {split: len(rows) for split, rows in split_rows.items()}
    if actual_hashes != frozen["split_identity_sha256"] or actual_sizes != frozen["split_sizes"]:
        raise ValueError("Formal training requires the supplied frozen EAV split identities. Use --smoke for a development manifest.")


def create_output_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {path}. Choose a fresh --output-dir.")
    path.mkdir(parents=True, exist_ok=False)


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def validate_training_config(config):
    if config.model != "residual_align_fmpa_v5" or config.run != "eav":
        raise ValueError("Only the complete residual_align_fmpa_v5 EAV experiment is supported")
    if config.temporal_slots != 16 or config.num_classes != len(LABELS):
        raise ValueError("The fixed EAV features require temporal_slots=16 and num_classes=5")
    if config.epochs < 1 or config.batch_size < 1 or config.eval_batch_size < 1 or config.workers < 0:
        raise ValueError("Epochs and batch sizes must be positive; workers must be nonnegative")
    if config.lr <= 0 or config.weight_decay < 0 or config.grad_clip <= 0 or not 0 <= config.dropout < 1:
        raise ValueError("Invalid optimizer/dropout settings")
    if any(not math.isfinite(getattr(config, name)) for name in (
        "lr", "weight_decay", "grad_clip", "dropout", "v5_residual_gate_init",
        "v5_alignment_aux_weight", "v5_fmpa_aux_weight",
    )):
        raise ValueError("Hyperparameters must be finite")
    if config.v5_alignment_aux_weight <= 0 or config.v5_fmpa_aux_weight <= 0:
        raise ValueError("The complete experiment requires both positive V5 auxiliary loss weights")
    if not config.smoke:
        changed = {key: getattr(config, key) for key, value in FORMAL_SETTINGS.items() if getattr(config, key) != value}
        if changed or config.limit_per_split != -1:
            raise ValueError(f"Formal settings are frozen. Use --smoke and a fresh output directory for development: {changed}")
    elif config.limit_per_split < 1:
        raise ValueError("Smoke runs require a positive --limit-per-split to keep them separate and small")
    if config.smoke and config.output_dir.resolve() == Config().output_dir.resolve():
        raise ValueError("Smoke runs require a separate --output-dir instead of the default formal output directory")


def add_config_arguments(parser):
    parser.add_argument("--config", type=Path, help="Optional JSON configuration; CLI values override it")
    for name in ("manifest", "features_root", "data_root", "output_dir", "stats_cache"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=None)
    for name in ("eeg_feature_name", "audio_feature_name", "video_feature_name", "device"):
        parser.add_argument("--" + name.replace("_", "-"), default=None)
    for name in ("epochs", "patience", "batch_size", "eval_batch_size", "workers", "seed", "token_dim", "hidden_dim", "num_heads", "temporal_slots", "v5_local_radius", "limit_per_split"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    for name in ("lr", "weight_decay", "dropout", "grad_clip", "v5_residual_gate_init", "v5_alignment_aux_weight", "v5_fmpa_aux_weight"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=None)
    parser.add_argument("--preload-features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--smoke", action="store_true", default=None, help="Small development run; outputs are explicitly marked as non-paper results")


def training_config_from_args(args):
    values = Config().to_dict()
    supplied = {}
    if args.config is not None:
        with args.config.open(encoding="utf-8") as handle:
            supplied = json.load(handle)
        # Validate all file-provided field names, including those later overridden.
        Config.from_dict(supplied)
        values.update(supplied)
    values.update({key: value for key, value in vars(args).items() if key in values and value is not None})
    explicit_features = args.features_root is not None or supplied.get("features_root") is not None
    if (args.data_root is not None or supplied.get("data_root") is not None) and not explicit_features:
        values["features_root"] = str(Path(values["data_root"]) / "eav_features")
    if values["smoke"] and args.limit_per_split is None and values["limit_per_split"] < 0:
        values["limit_per_split"] = 5
    if values["smoke"] and args.epochs is None and values["epochs"] == 60:
        values["epochs"] = 1
    config = Config.from_dict(values)
    validate_training_config(config)
    return config


def validate_feature_files(config, split_rows, splits=("train", "val", "test")):
    missing = []
    for split in splits:
        if not split_rows[split]:
            raise ValueError(f"The {split} split is empty")
        for row in split_rows[split]:
            for path in feature_paths(config.features_root, row, config.audio_feature_name, config.video_feature_name, config.eeg_feature_name):
                if not path.is_file():
                    missing.append(path)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} feature files; first={missing[0]}")


def validate_statistics_features(stats, config):
    metadata = stats["metadata"]
    if metadata.get("provenance_verified"):
        expected = {"eeg": config.eeg_feature_name, "audio": config.audio_feature_name, "video": config.video_feature_name}
        if metadata.get("feature_names") != expected:
            raise ValueError("Statistics were computed using different feature names")
        if metadata.get("feature_shapes") != {"eeg": [16, 150], "audio": [64, 90], "video": [64, 1408]}:
            raise ValueError("Statistics feature shapes do not match the fixed EAV features")


def build_loaders(config, split_rows, stats, device, splits=("train", "val", "test")):
    datasets = {split: EAVFeatureDataset(
        split_rows[split], config.features_root, config.audio_feature_name,
        config.video_feature_name, config.eeg_feature_name, (16, 150), stats,
        preload=config.preload_features, split_name=split,
    ) for split in splits}
    return {split: make_loader(
        dataset, config.batch_size if split == "train" else config.eval_batch_size,
        shuffle=split == "train", workers=config.workers,
        pin_memory=device.type == "cuda", seed=config.seed,
    ) for split, dataset in datasets.items()}


@torch.no_grad()
def evaluate_model(model, loader, device, amp_enabled):
    model.eval()
    targets, predictions, sample_ids = [], [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    for eeg, audio, video, target, _, ids in loader:
        eeg, audio, video, target = [value.to(device, non_blocking=True) for value in (eeg, audio, video, target)]
        active_mask = active_mask_for_run("eav", target.size(0), device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(eeg, audio, video, active_mask)
            loss = criterion(logits, target)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation logits/loss")
        total_loss += float(loss.detach().cpu()) * target.size(0)
        targets.extend(target.cpu().tolist())
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        sample_ids.extend(ids)
    if not targets:
        raise ValueError("Cannot evaluate an empty split")
    return {
        "loss": total_loss / len(targets), "acc": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(len(LABELS)))).tolist(),
        "labels": targets, "predictions": predictions, "sample_ids": sample_ids,
    }


def metrics_only(result):
    return {key: result[key] for key in ("loss", "acc", "macro_f1", "confusion_matrix")}


def save_predictions(path, result):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "true_label", "pred_label", "true_label_id", "pred_label_id", "correct"])
        writer.writeheader()
        for sample_id, truth, prediction in zip(result["sample_ids"], result["labels"], result["predictions"]):
            writer.writerow({"sample_id": sample_id, "true_label": LABELS[truth], "pred_label": LABELS[prediction], "true_label_id": truth, "pred_label_id": prediction, "correct": int(truth == prediction)})
