"""Train the complete EAV TAlign-FMPA V5 experiment, or a marked smoke run."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from data.dataset import (
    LABELS, active_mask_for_run, compute_train_stats, load_statistics,
    read_manifest, save_statistics, statistics_to_serializable, validate_statistics,
)
from model import TAlignFMPA
from utils.checkpoint import CHECKPOINT_FORMAT, CHECKPOINT_VERSION, cpu_state_dict
from utils.experiment import (
    add_config_arguments, build_loaders, create_output_directory, evaluate_model,
    metrics_only, resolve_device, save_json, save_predictions, set_seed,
    sha256_file, source_provenance, split_provenance, training_config_from_args,
    validate_feature_files, validate_formal_split_protocol, validate_statistics_features,
)


def train(config):
    """Keep the original seed, loader RNG, AdamW, AMP and strict F1 selection."""
    from utils.experiment import validate_training_config
    validate_training_config(config)
    if config.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {config.output_dir}")
    set_seed(config.seed)
    device = resolve_device(config.device)
    amp_enabled = device.type == "cuda"
    split_rows = read_manifest(config.manifest, config.limit_per_split)
    validate_formal_split_protocol(config, split_rows)
    validate_feature_files(config, split_rows)
    if config.stats_cache is None:
        config.stats_cache = config.output_dir / "train_stats.npz"
    if config.stats_cache.is_file():
        stats = load_statistics(config.stats_cache)
        stats = validate_statistics(stats, split_rows["train"], config.manifest)
        validate_statistics_features(stats, config)
        stats_source = {"kind": "verified_train_cache", "path": str(config.stats_cache)}
    else:
        print("Computing train-only EEG/audio standardization statistics...", flush=True)
        stats = compute_train_stats(
            split_rows["train"], config.features_root, config.audio_feature_name,
            config.video_feature_name, config.eeg_feature_name, (16, 150),
            manifest_path=config.manifest,
        )
        stats_source = {"kind": "computed_from_train_split", "path": str(config.stats_cache)}
    # Reserve a new output before producing any artifacts; never resume or overwrite.
    create_output_directory(config.output_dir)
    if not config.stats_cache.exists():
        save_statistics(stats, config.stats_cache)
    stats_source["cache_sha256"] = sha256_file(config.stats_cache)
    provenance = split_provenance(config.manifest, split_rows)
    sources = source_provenance()
    loaders = build_loaders(config, split_rows, stats, device)
    model = TAlignFMPA(config).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    # betas/eps/amsgrad/foreach/fused remain the PyTorch AdamW defaults, as before.
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_val, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []
    print(f"Device={device} AMP={amp_enabled} model={config.model} run=eav", flush=True)
    print("Split sizes: " + ", ".join(f"{key}={len(value)}" for key, value in split_rows.items()), flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in parameters):,}", flush=True)
    if config.smoke:
        print("SMOKE RUN: development verification only; these are not paper results.", flush=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = total_ce = total_alignment = total_fmpa = 0.0
        total_correct = total = 0
        for eeg, audio, video, target, _, _ in loaders["train"]:
            eeg, audio, video, target = [value.to(device, non_blocking=True) for value in (eeg, audio, video, target)]
            active_mask = active_mask_for_run("eav", target.size(0), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(eeg, audio, video, active_mask)
                ce_loss = criterion(logits, target)
                alignment_loss = logits.new_zeros(())
                fmpa_loss = logits.new_zeros(())
                if config.v5_alignment_aux_weight > 0:
                    if model.last_v5_alignment_aux_logits is None:
                        raise RuntimeError("Alignment auxiliary logits were not produced")
                    alignment_loss = criterion(model.last_v5_alignment_aux_logits, target)
                if config.v5_fmpa_aux_weight > 0:
                    if model.last_v5_fmpa_aux_logits is None:
                        raise RuntimeError("FMPA auxiliary logits were not produced")
                    fmpa_loss = criterion(model.last_v5_fmpa_aux_logits, target)
                loss = ce_loss + config.v5_alignment_aux_weight * alignment_loss + config.v5_fmpa_aux_weight * fmpa_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_size = target.size(0)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_ce += float(ce_loss.detach().cpu()) * batch_size
            total_alignment += float(alignment_loss.detach().cpu()) * batch_size
            total_fmpa += float(fmpa_loss.detach().cpu()) * batch_size
            total_correct += int((logits.argmax(dim=-1) == target).sum().detach().cpu())
            total += batch_size
        val = evaluate_model(model, loaders["val"], device, amp_enabled)
        row = {
            "epoch": epoch, "train_loss": total_loss / total, "train_ce_loss": total_ce / total,
            "train_v5_alignment_aux_loss": total_alignment / total,
            "train_v5_fmpa_aux_loss": total_fmpa / total, "train_acc": total_correct / total,
            "val_loss": val["loss"], "val_acc": val["acc"], "val_macro_f1": val["macro_f1"],
        }
        history.append(row)
        print(f"epoch={epoch:03d} train_acc={row['train_acc']:.4f} val_acc={val['acc']:.4f} val_macro_f1={val['macro_f1']:.4f}", flush=True)
        # Strict improvement selects the first occurrence of a tied best score.
        if val["macro_f1"] > best_val:
            best_val, best_epoch, best_state, stale = val["macro_f1"], epoch, cpu_state_dict(model), 0
        else:
            stale += 1
        if config.patience > 0 and stale >= config.patience:
            print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    selected_val = evaluate_model(model, loaders["val"], device, amp_enabled)
    # Test is first evaluated after validation has selected the checkpoint.
    test = evaluate_model(model, loaders["test"], device, amp_enabled)
    summary = {
        "experiment": "TAlign-FMPA V5 complete EAV", "model": config.model, "run": "eav",
        "smoke": config.smoke, "paper_result": False,
        "uses_paper_training_protocol": not config.smoke,
        "result_status": "smoke_only_not_for_paper" if config.smoke else "new_run_unverified_against_paper",
        "labels": list(LABELS), "label_to_id": {label: i for i, label in enumerate(LABELS)},
        "config": config.to_dict(), "actual_device": str(device), "amp_enabled": amp_enabled,
        "best_epoch": best_epoch, "best_val_macro_f1": best_val, "epochs_completed": len(history),
        "selection_rule": "strictly greater validation five-class Macro-F1; earliest tie wins",
        "early_stopping_patience": config.patience,
        "Acc": test["acc"], "MacroF1": test["macro_f1"],
        "test_acc": test["acc"], "test_macro_f1": test["macro_f1"],
        "test_confusion_matrix": test["confusion_matrix"],
        "validation": metrics_only(selected_val), "test": metrics_only(test),
        "source_provenance": sources, "split_provenance": provenance,
        "statistics_source": stats_source, "statistics_metadata": stats["metadata"],
    }
    checkpoint = {
        "format": CHECKPOINT_FORMAT, "format_version": CHECKPOINT_VERSION,
        "model": best_state, "config": config.to_dict(), "labels": list(LABELS),
        "stats": statistics_to_serializable(stats), "source_provenance": sources,
        "split_provenance": provenance, "summary": summary,
    }
    torch.save(checkpoint, config.output_dir / "best_model.pth")
    with (config.output_dir / "train_log.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    save_predictions(config.output_dir / "validation_predictions.csv", selected_val)
    save_predictions(config.output_dir / "test_predictions.csv", test)
    save_json(config.output_dir / "summary.json", summary)
    print(f"Selected epoch={best_epoch}; test Acc={test['acc']:.6f}, MacroF1={test['macro_f1']:.6f}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    config = training_config_from_args(args)
    train(config)


if __name__ == "__main__":
    main()
