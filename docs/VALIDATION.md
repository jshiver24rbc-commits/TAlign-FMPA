# Extraction and validation record

Prepared on 2026-11-19 for the EAV-only full-model release. This record
separates source equivalence, integration checks, and paper-score reproduction.

## Source identity

| Source file in Emotion-LLaMA-v2 | SHA-256 |
|---|---|
| `scripts/train_eav_clean_mlce_fmpa.py` | `26520026d07b291c39fc31a8fa90191ae7fe020ab688ed95dbaa7df909a7dc4d` |
| `scripts/run_eav_temporal_fmpa_v5_formal_seed42.sh` | `971f57811622f01691f9879d41c012cec36e0cbff55b69c009db8d7d7bd277a4` |
| `scripts/eav_public_fusion_models.py` (dependency reviewed; not required here) | `e15bcf2dcb600738594e5758545ffd874d73493121f711f735362b6406c892f6` |

The original training script also defines historical modules that the complete
V5 forward path does not use. Those modules are not part of this repository.
Active parameter names were retained to permit checkpoint migration.

Constructing the removed modules consumed random numbers in the original
implementation. `model/_formal_initialization.py` replays their initialization
draw schedule without retaining their network implementations. This preserves
active initialization and the CPU RNG state in the tested environment.
It is not a guarantee across PyTorch versions, devices, or default dtypes.

## Completed checks

| Check | Result / scope |
|---|---|
| Active model extraction | 126 parameter tensors; source initialization exactly equal |
| Constructor and forward RNG | Exactly equal in the tested CPU environment |
| Forward behavior | Train/eval main logits, auxiliary logits, attention/gate diagnostics exactly equal |
| Training update | Loss, all active gradients, and one gradient-clipped AdamW update exactly equal |
| Configurations checked | Formal dimensions with seeds 42/43; an additional small dimension configuration |
| Data and normalization | Source and extracted EEG/audio statistics, normalized tensors, unmodified video tensors, shuffle/collation equal |
| Public split | All 16,800 ordered identities/labels/subsets preserved; 0 cross-subset trial overlaps |
| Historical checkpoint | Actual formal C11 checkpoint migrated; CPU finite-logit forward passed |
| New checkpoint | Safe `weights_only=True` roundtrip passed |
| End-to-end integration | 15 synthetic samples, one epoch; independent evaluation with test features only |
| Integration result agreement | Metrics, label order, prediction CSV identical; checkpoint bytes unchanged |
| Negative cases | Missing/unknown/shape-mismatched parameters, altered statistics hash, unexpected split, and existing output directories rejected |

Model comparisons used local PyTorch `2.11.0+cpu`. Synthetic integration used a
smaller model to check execution only. Synthetic features and test artifacts
were kept outside this repository, and their metrics are not reported as
scientific results.

## Parameter accounting

The active full model contains 2,096,376 parameters, including 14,346 parameters
in two training auxiliary heads. Excluding those heads gives 2,082,030
parameters, consistent with the manuscript's rounded 2.082M main-path count.
The original monolithic object also registered unused historical networks;
its total registered parameter count is not the inference-path count.

## Not completed by this extraction

- Full formal-cache training on the server.
- Re-evaluation of all 5,040 real test windows with the extracted code.
- Numerical verification of the manuscript's 85.95% / 86.04% scores from a new run.
- Reconstruction of every original feature-extraction command and cache fingerprint.
- Validation of CUDA behavior or historical PyTorch 2.0 in this local environment.

The formal cache is not complete locally. Tensor shape checks cannot establish
cache equivalence. A server run using the matching cache and checkpoint is the
next validation step. Keep the original experimental artifacts unchanged and
write all new results to a fresh directory.
