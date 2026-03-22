# SKAM odometer fraud project
This project is all about creating a machine learning model to detect odometer fraud

## Notes
gitignore ignores the data folder, please put the data there, do not commit huge files to the repository

## Data
Links:
- Link for transeksta's dataset for technical inspections: https://get.data.gov.lt/datasets/gov/transeksta/ctadb/Apziura/:format/csv

## Inspection failure model
This repository includes a scalable training script that predicts failed technical inspection risk from:
- `tp_marke` + `tp_modelis` (vehicle label)
- `tp_rida_km` (mileage)
- `tp_pag_metai` (year)
- `tp_kuras` (fuel)
- `tp_dumingumas` (smokiness)

The script reads `data/Apziura.csv` in chunks, trains an incremental logistic model, and saves:
- model artifact: `models/inspection_failure_model.joblib`
- ranked high-risk pairs: `models/high_risk_model_mileage_pairs.csv`

The file is not stored in the repository, please download it from the link above and put it in the `data` folder.

### Setup
```bash
pip install -r requirements.txt
```

### Train
```bash
python models/train_inspection_failure_model.py \
	--csv-path data/Apziura.csv \
	--output-model models/inspection_failure_model.joblib \
	--output-risk-report models/high_risk_model_mileage_pairs.csv
```

Quick smoke test (process first 20 chunks only):
```bash
python models/train_inspection_failure_model.py --max-chunks 20
```

### Available parameters

The training script supports the following CLI arguments:

- `--csv-path` (default: `data/Apziura.csv`)
	Path to the input CSV file.
- `--output-model` (default: `models/inspection_failure_model.joblib`)
	Output path for the trained model artifact.
- `--output-risk-report` (default: `models/high_risk_model_mileage_pairs.csv`)
	Output path for the ranked risk CSV.
- `--chunk-size` (default: `100000`)
	Number of rows processed per chunk.
- `--max-eval-rows` (default: `150000`)
	Maximum holdout rows used for evaluation.
- `--min-model-count` (default: `1`)
	Minimum row count for a model to be included in risk ranking.
- `--top-model-limit` (default: `0`)
	Limit of most common models to score in the risk table. `0` means all.
- `--random-state` (default: `42`)
	Random seed for reproducible sampling.
- `--max-chunks` (default: `0`)
	Process only first N chunks. `0` means process all chunks.
- `--fbeta-beta` (default: `2.0`)
	Beta value for F-beta metric. Use values greater than `1` to favor recall.

Example with custom parameters:

```bash
python models/train_inspection_failure_model.py \
	--csv-path data/Apziura.csv \
	--output-model models/inspection_failure_model.joblib \
	--output-risk-report models/high_risk_model_mileage_pairs.csv \
	--chunk-size 120000 \
	--max-eval-rows 200000 \
	--max-chunks 0 \
	--fbeta-beta 2.0
```

### What the output means
- `predicted_failure_probability` close to `1.0` means the model expects a high chance of inspection failure.
- The risk report ranks `vehicle_label` and mileage combinations that are most likely to fail.