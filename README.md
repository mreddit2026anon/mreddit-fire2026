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

## Data placement

The scripts expect organizer-provided files under a local `data/raw/` directory. Do not commit these files. Expected filenames and task contracts are validated by the code before training or inference.

English and Tulu inputs use `title` and `text`; Kannada uses `Text`. Official test data must not be used for training, validation, feature fitting, early stopping, model selection, or error analysis.

## Environment setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install -r requirements-transformer.txt
```

For XLM-R experiments, install a PyTorch build compatible with the available CUDA environment before installing the Transformer dependencies.

## Read-only checks and smoke tests

Run the unit tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Validate the data-audit flow without writing audit outputs:

```bash
python scripts/audit_data.py --smoke
```

Check and smoke-test the traditional routes:

```bash
python scripts/train_cpu_baseline.py --check-inputs
python scripts/train_cpu_baseline.py --smoke
python scripts/run_phase_b_cv.py --check-inputs
python scripts/run_phase_b_cv.py --smoke
python scripts/finalize_phase_b.py --check-inputs
python scripts/finalize_phase_b.py --smoke
```

Check the Transformer environment and task inputs:

```bash
python scripts/check_gpu_env.py
python scripts/train_transformer.py --task english_domain --check-inputs
python scripts/train_transformer.py --task english_domain --smoke
```

Smoke tests should be completed before full-data or long-running commands. Different evaluation protocols in the paper serve different purposes and their scores should not be treated as a single directly comparable ranking.

## Configuration note

The configuration files preserve archived experiment parameters and status fields. They are not substitutes for missing trainer histories, checkpoint hashes, prediction artifacts, or a complete model-to-submission provenance chain. In particular, status fields should not be interpreted as a complete registry of all remote runs.

## Submission validation

Submission construction preserves the organizer-provided ID order and the `id,Regret,Domain` column contract. Kannada Domain cells must remain empty. Validate locally generated artifacts with:

```bash
python scripts/validate_submission.py --csv PATH_TO_SUBMISSION.csv
python scripts/validate_submission.py --zip PATH_TO_SUBMISSION.zip
```

No organizer data, official test predictions, or submitted archive is included in this repository.
