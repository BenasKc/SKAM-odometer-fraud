#!/usr/bin/env python3
"""Train a car inspection failure model using car model and mileage.

This script is designed for very large CSV files and trains incrementally
using chunked reads and SGD logistic regression.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, fbeta_score, roc_auc_score

THRESHOLD = 0.1

@dataclass
class TrainingMetrics:
    roc_auc: float
    f1: float
    fbeta: float
    beta: float
    report_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a model that predicts failed technical inspection risk "
            "from car model and mileage."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/Apziura.csv"),
        help="Path to the inspection CSV file.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("models/inspection_failure_model.joblib"),
        help="Where to save the trained model artifact.",
    )
    parser.add_argument(
        "--output-risk-report",
        type=Path,
        default=Path("models/high_risk_model_mileage_pairs.csv"),
        help="Where to save the ranked high-risk model and mileage pairs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per chunk while streaming CSV.",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=150_000,
        help="Maximum number of holdout rows used for final evaluation.",
    )
    parser.add_argument(
        "--min-model-count",
        type=int,
        default=1,
        help="Minimum number of rows for a model to be included in risk ranking.",
    )
    parser.add_argument(
        "--top-model-limit",
        type=int,
        default=0,
        help="Limit number of frequent models to include in risk scan (0 means all).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible holdout sampling.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help=(
            "Limit number of CSV chunks processed (0 means process all chunks). "
            "Useful for quick smoke tests."
        ),
    )
    parser.add_argument(
        "--fbeta-beta",
        type=float,
        default=2.0,
        help=(
            "Beta value for F-beta metric. Use beta > 1 to favor recall for failing cars. "
            "Example: 2.0."
        ),
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip().upper()
    return text if text else "UNKNOWN"


def build_vehicle_label(make_value: object, model_value: object) -> str:
    make = normalize_text(make_value)
    model = normalize_text(model_value)
    if make == "UNKNOWN" and model == "UNKNOWN":
        return "UNKNOWN"
    if make == "UNKNOWN":
        return model
    if model == "UNKNOWN":
        return make
    return f"{make} {model}"


def parse_passed_flag(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "t", "yes", "y"}
    return lowered.isin(true_values)


def preprocess_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "tp_marke",
        "tp_modelis",
        "tp_rida_km",
        "tp_pag_metai",
        "tp_kuras",
        "tp_dumingumas",
        "ar_ta_islaikyta",
    ]
    work = chunk[required_columns].copy()

    work["vehicle_label"] = [
        build_vehicle_label(make, model)
        for make, model in zip(work["tp_marke"], work["tp_modelis"])
    ]
    work["tp_rida_km"] = pd.to_numeric(work["tp_rida_km"], errors="coerce")
    work["tp_pag_metai"] = pd.to_numeric(work["tp_pag_metai"], errors="coerce")
    work["tp_dumingumas"] = pd.to_numeric(work["tp_dumingumas"], errors="coerce")
    work["tp_kuras"] = work["tp_kuras"].map(normalize_text)

    work = work.dropna(subset=["tp_rida_km", "ar_ta_islaikyta"])
    work = work[work["tp_rida_km"] >= 0]

    passed = parse_passed_flag(work["ar_ta_islaikyta"])
    work["failed"] = (~passed).astype(np.int8)
    return work[
        [
            "vehicle_label",
            "tp_rida_km",
            "tp_pag_metai",
            "tp_kuras",
            "tp_dumingumas",
            "failed",
        ]
    ]


def to_feature_dicts(
    models: Iterable[str],
    mileages: Iterable[float],
    years: Iterable[float],
    fuels: Iterable[str],
    smokiness_values: Iterable[float],
) -> list[dict[str, float]]:
    # Mileage is rescaled for numeric stability with linear models.
    feature_rows: list[dict[str, float]] = []
    for model, mileage, year, fuel, smokiness in zip(
        models,
        mileages,
        years,
        fuels,
        smokiness_values,
    ):
        row = {
            f"model={model}": 1.0,
            f"fuel={fuel}": 1.0,
            "mileage_scaled": float(mileage) / 100_000.0,
        }

        if pd.notna(year):
            row["year_centered"] = (float(year) - 2000.0) / 30.0
        else:
            row["year_missing"] = 1.0

        if pd.notna(smokiness):
            row["smokiness_scaled"] = float(smokiness) / 100.0
        else:
            row["smokiness_missing"] = 1.0

        feature_rows.append(row)

    return feature_rows


def train_incremental_model(
    args: argparse.Namespace,
) -> tuple[FeatureHasher, SGDClassifier, Counter, dict[str, dict[str, float | str]], TrainingMetrics]:
    hasher = FeatureHasher(n_features=2**18, input_type="dict", alternate_sign=False)
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        random_state=args.random_state,
        average=True,
    )

    rng = np.random.default_rng(args.random_state)
    model_counts: Counter = Counter()
    model_fuel_counts: dict[str, Counter] = defaultdict(Counter)
    model_year_sums: dict[str, float] = defaultdict(float)
    model_year_counts: dict[str, int] = defaultdict(int)
    model_smoke_sums: dict[str, float] = defaultdict(float)
    model_smoke_counts: dict[str, int] = defaultdict(int)

    eval_features: list[dict[str, float]] = []
    eval_labels: list[int] = []

    total_rows = 0
    trained_rows = 0

    chunks = pd.read_csv(
        args.csv_path,
        usecols=[
            "tp_marke",
            "tp_modelis",
            "tp_rida_km",
            "tp_pag_metai",
            "tp_kuras",
            "tp_dumingumas",
            "ar_ta_islaikyta",
        ],
        chunksize=args.chunk_size,
        low_memory=True,
    )

    first_fit_done = False

    for chunk_idx, chunk in enumerate(chunks, start=1):
        if args.max_chunks > 0 and chunk_idx > args.max_chunks:
            break

        processed = preprocess_chunk(chunk)
        if processed.empty:
            continue

        n_rows = len(processed)
        total_rows += n_rows
        model_counts.update(processed["vehicle_label"].tolist())

        for model, fuel in zip(processed["vehicle_label"], processed["tp_kuras"]):
            model_fuel_counts[model][fuel] += 1

        year_rows = processed.dropna(subset=["tp_pag_metai"])[["vehicle_label", "tp_pag_metai"]]
        for model, year in year_rows.itertuples(index=False):
            model_year_sums[model] += float(year)
            model_year_counts[model] += 1

        smoke_rows = processed.dropna(subset=["tp_dumingumas"])[["vehicle_label", "tp_dumingumas"]]
        for model, smoke in smoke_rows.itertuples(index=False):
            model_smoke_sums[model] += float(smoke)
            model_smoke_counts[model] += 1

        eval_quota_remaining = max(0, args.max_eval_rows - len(eval_labels))
        if eval_quota_remaining > 0:
            eval_fraction = min(0.12, eval_quota_remaining / max(n_rows, 1))
            eval_mask = rng.random(n_rows) < eval_fraction
        else:
            eval_mask = np.zeros(n_rows, dtype=bool)

        train_mask = ~eval_mask

        if eval_mask.any():
            eval_chunk = processed.loc[eval_mask]
            eval_features.extend(
                to_feature_dicts(
                    eval_chunk["vehicle_label"],
                    eval_chunk["tp_rida_km"],
                    eval_chunk["tp_pag_metai"],
                    eval_chunk["tp_kuras"],
                    eval_chunk["tp_dumingumas"],
                )
            )
            eval_labels.extend(eval_chunk["failed"].astype(int).tolist())

        if train_mask.any():
            train_chunk = processed.loc[train_mask]
            x_train = hasher.transform(
                to_feature_dicts(
                    train_chunk["vehicle_label"],
                    train_chunk["tp_rida_km"],
                    train_chunk["tp_pag_metai"],
                    train_chunk["tp_kuras"],
                    train_chunk["tp_dumingumas"],
                )
            )
            y_train = train_chunk["failed"].astype(int).to_numpy()

            # Approximate balanced training for streamed chunks.
            # This keeps rare failure rows influential without loading full data.
            pos_count = int((y_train == 1).sum())
            neg_count = int((y_train == 0).sum())
            total = max(pos_count + neg_count, 1)
            if pos_count > 0 and neg_count > 0:
                weight_pos = total / (2.0 * pos_count)
                weight_neg = total / (2.0 * neg_count)
                sample_weight = np.where(y_train == 1, weight_pos, weight_neg)
            else:
                sample_weight = np.ones_like(y_train, dtype=float)

            if not first_fit_done:
                classifier.partial_fit(
                    x_train,
                    y_train,
                    classes=np.array([0, 1]),
                    sample_weight=sample_weight,
                )
                first_fit_done = True
            else:
                classifier.partial_fit(x_train, y_train, sample_weight=sample_weight)

            trained_rows += len(train_chunk)

        if chunk_idx % 20 == 0:
            print(
                f"Processed chunks: {chunk_idx}, rows seen: {total_rows:,}, "
                f"rows used for training: {trained_rows:,}, holdout size: {len(eval_labels):,}"
            )

    if not first_fit_done:
        raise RuntimeError("Training did not run; no usable rows found in dataset.")

    if not eval_labels:
        raise RuntimeError("No evaluation rows collected. Increase --max-eval-rows.")

    x_eval = hasher.transform(eval_features)
    y_eval = np.array(eval_labels)

    fail_scores = classifier.predict_proba(x_eval)[:, 1]
    roc_auc = roc_auc_score(y_eval, fail_scores)

    y_pred = (fail_scores >= THRESHOLD).astype(int)
    f1 = f1_score(y_eval, y_pred, zero_division=0)
    fbeta = fbeta_score(y_eval, y_pred, beta=args.fbeta_beta, zero_division=0)
    report_text = classification_report(y_eval, y_pred, digits=4)

    model_profiles: dict[str, dict[str, float | str]] = {}
    for model, count in model_counts.items():
        if count <= 0:
            continue
        top_fuel = model_fuel_counts[model].most_common(1)[0][0] if model_fuel_counts[model] else "UNKNOWN"
        avg_year = (
            model_year_sums[model] / model_year_counts[model]
            if model_year_counts[model] > 0
            else np.nan
        )
        avg_smokiness = (
            model_smoke_sums[model] / model_smoke_counts[model]
            if model_smoke_counts[model] > 0
            else np.nan
        )
        model_profiles[model] = {
            "fuel": top_fuel,
            "year": float(avg_year) if pd.notna(avg_year) else np.nan,
            "smokiness": float(avg_smokiness) if pd.notna(avg_smokiness) else np.nan,
        }

    metrics = TrainingMetrics(
        roc_auc=roc_auc,
        f1=f1,
        fbeta=fbeta,
        beta=args.fbeta_beta,
        report_text=report_text,
    )
    return hasher, classifier, model_counts, model_profiles, metrics


def build_risk_ranking(
    hasher: FeatureHasher,
    classifier: SGDClassifier,
    model_counts: Counter,
    model_profiles: dict[str, dict[str, float | str]],
    min_model_count: int,
    top_model_limit: int,
) -> pd.DataFrame:
    mileage_grid = [50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 400_000]

    if top_model_limit > 0:
        ranked_models = model_counts.most_common(top_model_limit)
    else:
        ranked_models = model_counts.most_common()

    candidate_models = [
        model
        for model, count in ranked_models
        if count >= min_model_count
    ]

    records = []
    for model in candidate_models:
        profile = model_profiles.get(model, {})
        default_fuel = str(profile.get("fuel", "UNKNOWN"))
        default_year = profile.get("year", np.nan)
        default_smokiness = profile.get("smokiness", np.nan)
        for mileage in mileage_grid:
            features = to_feature_dicts(
                [model],
                [mileage],
                [default_year],
                [default_fuel],
                [default_smokiness],
            )
            x = hasher.transform(features)
            fail_probability = float(classifier.predict_proba(x)[0, 1])
            records.append(
                {
                    "tp_modelis": model,
                    "tp_rida_km": mileage,
                    "predicted_failure_probability": round(fail_probability, 6),
                    "model_row_count": model_counts[model],
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "tp_modelis",
                "tp_rida_km",
                "predicted_failure_probability",
                "model_row_count",
            ]
        )

    risk_df = pd.DataFrame(records).sort_values(
        by=["predicted_failure_probability", "model_row_count"],
        ascending=[False, False],
    )
    return risk_df.reset_index(drop=True)


def main() -> None:
    args = parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_risk_report.parent.mkdir(parents=True, exist_ok=True)

    hasher, classifier, model_counts, model_profiles, metrics = train_incremental_model(args)

    artifact = {
        "feature_hasher": hasher,
        "classifier": classifier,
        "metadata": {
            "input_columns": [
                "tp_marke",
                "tp_modelis",
                "tp_rida_km",
                "tp_pag_metai",
                "tp_kuras",
                "tp_dumingumas",
            ],
            "target_column": "failed",
            "fail_definition": "failed = not ar_ta_islaikyta",
            "model_type": "SGDClassifier(log_loss)",
            "decision_threshold": THRESHOLD,
        },
    }
    joblib.dump(artifact, args.output_model)

    risk_df = build_risk_ranking(
        hasher,
        classifier,
        model_counts,
        model_profiles,
        min_model_count=args.min_model_count,
        top_model_limit=args.top_model_limit,
    )
    risk_df.to_csv(args.output_risk_report, index=False)

    print("\nTraining complete.")
    print(f"ROC AUC (holdout): {metrics.roc_auc:.4f}")
    print(f"F1 score (holdout): {metrics.f1:.4f}")
    print(f"F-beta score (holdout, beta={metrics.beta:.2f}): {metrics.fbeta:.4f}")
    print("Classification report:")
    print(metrics.report_text)

    print(f"Model saved to: {args.output_model}")
    print(f"Risk ranking saved to: {args.output_risk_report}")


if __name__ == "__main__":
    main()
