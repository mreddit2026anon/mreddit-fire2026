# Anonymous mReDDIT FIRE 2026 System Code

This repository contains an anonymous code release accompanying a participant paper for the mReDDIT shared task at FIRE 2026. It covers Regret Detection in English, Tulu, and Kannada, and Domain Identification in English and Tulu. Kannada Domain Identification is not part of the task and must remain empty in any submission.

Regret Detection uses the labels `Action`, `Inaction`, and `No Regret`. Domain Identification uses `Education`, `Health`, `Career and Finance`, `Romance friends and parents`, and `Other domains`.

## Scope

The release includes code for the traditional sparse-model route, Phase B candidate evaluation, independent XLM-R experiments, prediction generation, and validated task-level submission construction. The final submission assigns one existing prediction source to each valid task; it is not an ensemble, voting system, probability fusion method, jointly trained model, or new architecture.

Raw data, official test data, prediction files, submissions, model weights, checkpoints, and training logs are not distributed. Users must obtain the task data from the organizers under the applicable access terms. This code release does not claim complete run-level or end-to-end provenance.

## Candidate_B task-level source assignment

Candidate_B selects one already evaluated prediction source for each valid language--task combination:

| Language and task | Prediction source |
| --- | --- |
| English Regret Detection | XLM-R |
| English Domain Identification | XLM-R |
| Tulu Regret Detection | Character TF-IDF with balanced one-vs-rest logistic regression |
| Tulu Domain Identification | Phase A character TF-IDF with balanced LinearSVC |
| Kannada Regret Detection | Phase A traditional prediction source |
| Kannada Domain Identification | Not applicable; the output must remain empty |

These task-level outputs are selected as complete prediction columns and are not combined at the probability or decision level.

## Repository structure

- `src/`: shared data, traditional-model, Transformer-data, and submission utilities.
- `scripts/`: data auditing, training, evaluation, inference, and submission-validation entry points.
- `configs/`: archived experiment and submission configurations. Environment-specific absolute paths have been removed from this anonymous copy.
- `tests/`: unit tests based on synthetic or temporary inputs.
- `requirements.txt`: traditional sparse-model and analysis dependencies.
- `requirements-transformer.txt`: additional Transformer dependencies. Install a compatible PyTorch/CUDA build separately for GPU experiments.

## Running the Released Pipeline

These commands were checked against the released scripts and configurations, not executed as an end-to-end test of this release. Start in the root of this public repository. The commands below use Bash; run them in order in the same shell and stop if a check fails. Use a fresh working copy for a new run. Input checks and smoke tests can write local logs, split manifests, or other artifacts.

On Windows, use Bash inside WSL to run the examples unchanged, including virtual-environment activation and the task loops. Create the environment inside WSL; do not reuse a native Windows virtual environment.

### 1. CPU environment and quick check (no organizer data or GPU)

Use Python 3.12 (the archived Transformer environment used 3.12.11). Create and activate the environment before installing packages:

```bash
set -e
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_mixed_submission.py --candidate candidate_b_english_transformer --dry-run
python -m unittest discover -s tests -p "test_*.py"
```

For native Windows PowerShell, the activation command is `.\.venv\Scripts\Activate.ps1` (the loops below are Bash). The dry run prints the configured prediction sources without loading a model, reading predictions, or creating a submission. `scripts/audit_data.py` checks the current working directory for both an `AGENTS.md` file and an existing `.git` entry, even in smoke mode. The release does not include `AGENTS.md`; omit this optional development audit. The pipeline below does not require it.

### 2. Obtain the official inputs

Request the original mReDDIT data and baseline submission template from the organizers via the [official task page](https://www.codabench.org/competitions/16713/) under their access terms. If historical downloads are unavailable, contact the organizers rather than substituting a different dataset or constructing a template. Place the files as follows:

```text
data/raw/training_data/
    english_train_data.csv
    tulu_train_data.csv
    kannada_train_data.csv
    kannada_validation_data.csv
data/raw/test_data/
    english_test_data.csv
    tulu_test_data.csv
    kannada_test_data.csv
official/
    baseline_predictions.csv
```

Keep the original bytes, IDs, columns and row order. English and Tulu use `title` and `text`; Kannada uses `Text`. The template defines the submission structure and ID order; it is not test ground truth. Do not use official test inputs for fitting or tuning, or redistribute the inputs with the code.

### 3. CPU route: generate the traditional predictions

Phase A evaluates and finalizes five task models. Generate its test predictions before moving on: Phase B finalization uses Phase A model metadata and compares predictions with Phase A outputs.

```bash
tasks=(english_regret english_domain tulu_regret tulu_domain kannada_regret)
python scripts/train_cpu_baseline.py --check-inputs
python scripts/train_cpu_baseline.py --smoke
for t in "${tasks[@]}"; do
  python scripts/train_cpu_baseline.py --task "$t"
  python scripts/train_cpu_baseline.py --finalize "$t"
done
python scripts/train_cpu_baseline.py --write-report
python scripts/predict_cpu_baseline.py
python scripts/make_submission.py  # Phase A baseline, not Candidate_B
```

Run Phase B cross-validation and summarize it **before** the finalization input check. The check verifies the raw-file hashes, required Phase A artifacts, and the winners and Macro-F1 means in `reports/tables/phase_b_cv_metrics.csv` against the archived configuration. If the evidence differs, stop and report the discrepancy; do not edit the metrics or disable the check.

```bash
python scripts/run_phase_b_cv.py --check-inputs
python scripts/run_phase_b_cv.py --smoke
python scripts/run_phase_b_cv.py --run
python scripts/run_phase_b_cv.py --summarize
python scripts/finalize_phase_b.py --check-inputs
python scripts/finalize_phase_b.py --smoke
for t in "${tasks[@]}"; do
  python scripts/finalize_phase_b.py --task "$t"
done
python scripts/finalize_phase_b.py --predict
```

### 4. GPU route: generate the two English XLM-R predictions

The archived environment records Python 3.12.11 and `torch==2.9.1+cu130`, but not a verified installation command. In the activated environment, follow the [official PyTorch installation instructions for previous versions](https://pytorch.org/get-started/previous-versions/#v291) (v2.9.1, Linux or Windows, CUDA 13.0) before installing the Transformer requirements below. The environment check compares exact package versions and requires CUDA and BF16 support. Training needs the base model and tokenizer from [FacebookAI/xlm-roberta-base (official model card)](https://huggingface.co/FacebookAI/xlm-roberta-base), available from a local cache or by download.

```bash
python -m pip install -r requirements-transformer.txt
python scripts/check_gpu_env.py
for t in english_domain english_regret; do
  python scripts/train_transformer.py --task "$t" --check-inputs
  python scripts/train_transformer.py --task "$t" --smoke
  python scripts/train_transformer.py --task "$t"
  python scripts/predict_transformer.py --task "$t"
done
```

Formal training requires input-check and smoke records for the same configuration. Prediction loads the newly trained model and tokenizer from `models/transformer_baseline/<task>/best/`. The Tulu XLM-R experiments were candidates only: they are not required to build Candidate_B, and their predictions are not used in it.

#### Optional: Tulu XLM-R Candidate Experiments

To rerun the two independent Tulu candidates with the default XLM-R configuration, use the same GPU environment and data layout as above. Skip this entire subsection if building Candidate_B: its Tulu Regret and Tulu Domain predictions come from the traditional sources in the assignment table.

```bash
for t in tulu_regret tulu_domain; do
  python scripts/train_transformer.py --task "$t" --check-inputs
  python scripts/train_transformer.py --task "$t" --smoke
  python scripts/train_transformer.py --task "$t"
done
```

Stop here for development experiments. Only if you also need test predictions for these candidates, use their saved best models and then validate the generated files:

```bash
for t in tulu_regret tulu_domain; do
  python scripts/predict_transformer.py --task "$t"
  python scripts/predict_transformer.py --task "$t" --validate-only
done
```

### 5. Build and validate Candidate_B

Run this only after both prediction routes finish. The builder reads the two English prediction CSVs from `outputs/transformer_baseline/predictions/` and the Tulu Regret, Tulu Domain and Kannada Regret CSVs from `outputs/phase_b_final/predictions/`. The latter is a staging directory: Tulu Domain and Kannada Regret retain their Phase A sources. See the task-by-task assignments above.

```bash
candidate=candidate_b_english_transformer
python scripts/build_mixed_submission.py --candidate "$candidate"
python scripts/build_mixed_submission.py --candidate "$candidate" --validate-only
out="outputs/transformer_mixed_submissions/$candidate"
python scripts/validate_submission.py --csv "$out/submission.csv" --zip "$out/submission.zip"
```

### Reproducibility boundary

These commands describe a **new run**, not an exact replay of official submission **850840**. Historical checkpoints, prediction files and the submitted ZIP are not distributed. Configuration status fields are not a complete record of historical runs. Local validation checks the package, IDs, row order and labels (including empty Kannada Domain); it does not compute official test Macro-F1 or prove that regenerated predictions match the historical submission.
