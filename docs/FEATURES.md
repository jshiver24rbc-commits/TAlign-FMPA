# Pre-extracted EAV feature inputs

This release consumes pre-extracted NumPy feature files. It does not distribute
EAV recordings, feature tensors, original predictions or trained weights. Obtain
the dataset and feature inputs under their applicable access and usage terms.

The complete extraction provenance and exact reproducible extractor for the
original formal cache have not been established. The feature names and shapes
below define the model input contract; they do not establish that a newly
extracted cache will reproduce the original features or reported scores.

## Directory and tensor contract

Point `features_root` at a private directory containing:

```text
eav_features/
  EEG-TDE-16/<sample_id>.npy
  audio-frame-lld/<sample_id>.npy
  EVA-ViT-G-5s/<sample_id>.npy
```

For example, the manifest identity `subject01_002_seg00` maps to one `.npy` file
with that basename in each directory. No recording paths are needed by the
loader. Do not substitute similarly named smoke or older feature directories.

| Input | Shape per sample | Runtime treatment |
|---|---|---|
| EEG | `(16, 150)` | Train-only per-feature standardization |
| Audio | `(64, 90)` | Train-only per-feature standardization |
| Video | `(64, 1408)` | Values unchanged after float32 conversion |

Each file is loaded with `numpy.load(allow_pickle=False)`, converted to float32,
then squeezed, matching the original formal loader. The resulting exact shape
and finite values are required. Shape compatibility alone does not demonstrate
equivalence to the original extractor.

The dataset returns `(eeg, audio, video, label, subject_id, sample_id)`. Default
PyTorch collation stacks the five tensor fields and collects the sample IDs.
Model inputs use EEG, Audio, Video order. The modality activity mask uses
**Video, Audio, EEG** order. Class IDs are fixed as `neutral=0`, `anger=1`,
`calm=2`, `sad=3`, `happy=4`.

## Normalization and sampling

Only training rows determine normalization statistics. EEG vectors from all 16
slots are concatenated and their population mean/std are computed in float64;
EEG std is clipped to at least `1e-6`. Audio sums and squared sums are accumulated
in float64 over all 64 rows of every training sample, then std is
`sqrt(max(E[x²] - E[x]², 1e-6))`. The four resulting vectors are saved as
float32. These formulas intentionally preserve the original difference between
the EEG and audio variance floors. Validation and test samples reuse those
training statistics. Video is not standardized.

The portable statistics cache is a non-pickled `.npz` containing `eeg_mean`,
`eeg_std`, `audio_mean`, `audio_std` and `metadata_json`. Metadata records the
manifest file SHA-256, ordered training identity/sample ID hashes, feature
names/shapes, sample count and statistics content hash. Statistics embedded in
a checkpoint can be used directly at evaluation time without accessing any
training feature files. Legacy caches without provenance can be explicitly
accepted for migration, but remain marked unverified; supplying a manifest does
not retroactively verify their origin.

The formal training loader uses PyTorch random shuffling, each sample once per
epoch, no replacement and no dropped final batch. Validation and test loaders
preserve manifest order. There is no weighted or subject-balanced sampling in
the formal run. Seed all RNGs before creating/iterating loaders to preserve the
original RNG consumption; no independent loader generator is introduced here.

## Fixed split and sanitization audit

The public CSV retains only `sample_id`, `subject_id`, `trial_id`, `trial_index`,
`segment_id`, `emotion`, `split`. Source row order and every retained value are
identical after export; recording paths were removed. CSV line endings are LF.

The filename contains `subject_wise`, but this protocol splits **trials within
each subject and emotion group**. All 42 subjects occur in all three sets. It
does not measure generalization to unseen subjects. A trial is identified by
`(subject_id, trial_index)`; all four five-second segments stay in the same set.

| Split | Samples | Trials | Subjects | Samples per emotion |
|---|---:|---:|---:|---:|
| Train | 10,080 | 2,520 | 42 | 2,016 |
| Validation | 1,680 | 420 | 42 | 336 |
| Test | 5,040 | 1,260 | 42 | 1,008 |
| Total | 16,800 | 4,200 | 42 distinct | 3,360 |

The export audit checked unique sample IDs, exactly segment IDs 0–3 for each
trial, zero trial overlap across sets, and exact source-to-export identity,
label and split equality. `read_manifest` rechecks duplicate sample/segment IDs,
known labels, known splits and trial leakage before applying any smoke limit.

Original filename: `eav_manifest_subject_wise_4x5s_60_10_30_seed42.csv`

Original SHA-256:
`f41f8c089370de2211ef5956eb53b56dc37cdb79f80de321a02322dd160919ac`

Sanitized public file: `splits/eav_manifest_subject_wise_4x5s_60_10_30_seed42.csv`

Sanitized SHA-256:
`bcdf6c4f34fbdfed419c78b966e7b2e53ccc611d0b332458d45525751525b489`

The byte hashes differ because path columns were removed and line endings were
made portable. A cache/checkpoint created for the sanitized manifest must record
its sanitized hash; the original byte hash is documented only as provenance.
An old statistics cache cannot be re-certified merely by changing its stored
manifest hash.

At the time of repository extraction, the local source tree had incomplete or
older caches and smoke directories, but did not contain all three complete
formal directories above. No full-feature training or reproduction of reported
scores was run as part of this release preparation.
