"""Evaluate a selected EAV V5 checkpoint using its train-only statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import Config
from data.dataset import LABELS, load_statistics, read_manifest, validate_statistics
from model import TAlignFMPA
from utils.checkpoint import checkpoint_config, load_active_model_state, load_checkpoint
from utils.experiment import (
    build_loaders, create_output_directory, evaluate_model, metrics_only,
    resolve_device, save_json, save_predictions, set_seed, sha256_file,
    source_provenance, split_provenance, validate_feature_files,
    validate_statistics_features,
)


def _verify_legacy_manifest(checkpoint, manifest):
    expected = checkpoint.get("summary", {}).get("manifest_sha256")
    actual = sha256_file(manifest)
    if expected is None:
        return {"verified": False, "reason": "historical_checkpoint_has_no_manifest_hash"}
    if expected == actual:
        return {"verified": True, "kind": "original_manifest_sha256"}
    # The public manifest removes private path columns. Its checked-in provenance
    # binds the original file hash to unchanged ordered identities in every split.
    bridge_path = Path(manifest).with_suffix(".provenance.json")
    if bridge_path.is_file():
        with bridge_path.open(encoding="utf-8") as handle:
            bridge = json.load(handle)
        current = split_provenance(manifest, read_manifest(manifest))
        hashes = {split: value["identity_sha256"] for split, value in current["splits"].items()}
        if (bridge.get("original_sha256") == expected
                and bridge.get("sanitized_sha256") == actual
                and bridge.get("split_identity_sha256") == hashes
                and bridge.get("original_identity_sha256")
                and bridge.get("original_identity_sha256") == bridge.get("sanitized_identity_sha256")):
            return {"verified": True, "kind": "sanitized_manifest_identity_bridge", "bridge_sha256": sha256_file(bridge_path), "original_manifest_sha256": expected}
    raise ValueError("Manifest differs from the historical checkpoint's training manifest; supply the original manifest or its verified sanitized manifest/provenance pair")


def _statistics_for_evaluation(checkpoint, config, split_rows, *, legacy):
    embedded = checkpoint.get("stats")
    if embedded is None and config.stats_cache is None:
        raise ValueError(
            "This checkpoint has no train statistics. Supply its original train-only "
            "statistics with --stats-cache; evaluation never recomputes statistics."
        )
    stats_source = {"kind": "checkpoint", "provenance_verified": False}
    stats = None
    if embedded is not None:
        stats = validate_statistics(
            embedded, split_rows["train"], config.manifest, allow_legacy=legacy,
        )
    if config.stats_cache is not None:
        cached = load_statistics(config.stats_cache, allow_legacy=legacy)
        cached = validate_statistics(cached, split_rows["train"], config.manifest, allow_legacy=legacy)
        cache_hash = sha256_file(config.stats_cache)
        if stats is not None:
            for key in ("eeg_mean", "eeg_std", "audio_mean", "audio_std"):
                if not np.array_equal(stats[key], cached[key]):
                    raise ValueError(f"--stats-cache does not match checkpoint train statistics: {key}")
        elif legacy:
            expected_hash = checkpoint.get("summary", {}).get("stats_cache_sha256")
            if expected_hash is not None and expected_hash != cache_hash:
                raise ValueError("--stats-cache SHA256 differs from the historical checkpoint's original train cache")
        stats = cached
        stats_source = {"kind": "explicit_stats_cache", "path": str(config.stats_cache), "sha256": cache_hash}
    validate_statistics_features(stats, config)
    stats_source["provenance_verified"] = bool(stats["metadata"].get("provenance_verified"))
    if not legacy and not stats_source["provenance_verified"]:
        raise ValueError("Modern checkpoints require verified train-only statistics")
    return stats, stats_source


def evaluate_checkpoint(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {args.output_dir}")
    checkpoint_hash = sha256_file(args.checkpoint)
    checkpoint, legacy = load_checkpoint(args.checkpoint, trusted_legacy=args.trusted_legacy_checkpoint)
    stored = checkpoint_config(checkpoint, legacy=legacy)
    defaults = Config().to_dict()
    if legacy:
        # Historical paths describe another machine; use this release's portable
        # locations unless the caller explicitly provides locations below.
        path_fields = {"manifest", "features_root", "data_root", "output_dir", "stats_cache"}
        values = {**defaults, **{key: value for key, value in stored.items() if key in defaults and key not in path_fields}}
        values["smoke"] = stored.get("limit_per_split", -1) >= 0
    else:
        if list(checkpoint["labels"]) != list(LABELS):
            raise ValueError("Checkpoint label order differs from the fixed EAV order")
        values = dict(stored)
        # A self-contained checkpoint always provides its own statistics; do not
        # implicitly follow the training machine's optional cache path.
        values["stats_cache"] = None
    for key in ("manifest", "features_root", "device", "workers", "eval_batch_size", "preload_features", "stats_cache", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    config = Config.from_dict(values)
    if config.eval_batch_size < 1 or config.workers < 0:
        raise ValueError("Evaluation batch size must be positive; workers must be nonnegative")
    set_seed(config.seed)
    device = resolve_device(config.device)
    split_rows = read_manifest(config.manifest, config.limit_per_split)
    provenance = split_provenance(config.manifest, split_rows)
    if legacy:
        manifest_verification = _verify_legacy_manifest(checkpoint, config.manifest)
    else:
        if provenance != checkpoint["split_provenance"]:
            raise ValueError("Manifest/split identities do not match this checkpoint")
        manifest_verification = {"verified": True, "kind": "checkpoint_split_and_manifest_hashes"}
    stats, stats_source = _statistics_for_evaluation(checkpoint, config, split_rows, legacy=legacy)
    # Evaluation only needs held-out features. Training features are never read.
    validate_feature_files(config, split_rows, splits=("test",))
    loaders = build_loaders(config, split_rows, stats, device, splits=("test",))
    model = TAlignFMPA(config).to(device)
    state_report = load_active_model_state(model, checkpoint["model"], legacy=legacy)
    result = evaluate_model(model, loaders["test"], device, device.type == "cuda")
    # A read-only evaluation must leave even historical checkpoint bytes intact.
    if sha256_file(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during evaluation")
    create_output_directory(config.output_dir)
    summary = {
        "experiment": "TAlign-FMPA V5 complete EAV", "model": config.model, "run": "eav",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
        "legacy_checkpoint": legacy, "checkpoint_state": state_report,
        "smoke": bool(config.smoke), "paper_result": False,
        "uses_paper_training_protocol": not bool(config.smoke),
        "result_status": "smoke_only_not_for_paper" if config.smoke else "checkpoint_evaluation_unverified_against_paper",
        "labels": list(LABELS), "label_to_id": {label: i for i, label in enumerate(LABELS)},
        "Acc": result["acc"], "MacroF1": result["macro_f1"],
        "test_acc": result["acc"], "test_macro_f1": result["macro_f1"],
        "test_confusion_matrix": result["confusion_matrix"], "test": metrics_only(result),
        "best_epoch": checkpoint.get("summary", {}).get("best_epoch"),
        "config": config.to_dict(), "actual_device": str(device), "amp_enabled": device.type == "cuda",
        "manifest_verification": manifest_verification, "split_provenance": provenance,
        "statistics_source": stats_source, "statistics_metadata": stats["metadata"],
        "checkpoint_source_provenance": checkpoint.get("source_provenance", {
            "historical_training_script_sha256": checkpoint.get("summary", {}).get("training_script_sha256"),
        }),
        "evaluation_source_provenance": source_provenance(),
    }
    save_predictions(config.output_dir / "test_predictions.csv", result)
    save_json(config.output_dir / "summary.json", summary)
    print(f"Test Acc={result['acc']:.6f}, MacroF1={result['macro_f1']:.6f}; labels={list(LABELS)}", flush=True)
    if config.smoke:
        print("SMOKE RUN: development verification only; these are not paper results.", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--features-root", type=Path)
    parser.add_argument("--stats-cache", type=Path, help="Original train-only cache; required if the checkpoint has no statistics")
    parser.add_argument("--trusted-legacy-checkpoint", action="store_true", help="Explicitly trust historical pickle content, including Namespace and Linux paths")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--preload-features", action=argparse.BooleanOptionalAction, default=None)
    return evaluate_checkpoint(parser.parse_args(argv))


if __name__ == "__main__":
    main()
