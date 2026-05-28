#!/usr/bin/env python3
"""Predict purchase risk for a single used-car listing.

Usage (interactive prompts):
    python predict_car_risk.py

Usage (all flags):
    python predict_car_risk.py \
        --make "Opel" --model "Astra" \
        --year 2012 --mileage 185000 \
        --engine-cc 1600 --horsepower 115 \
        --fuel "Petrol" --country "Lithuania" \
        --has-defects --ti-months 8 \
        --asking-price 2500
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SCORING_YEAR = 2026
MODELS_DIR   = Path(__file__).parent

PRICE_MODEL_PATH    = MODELS_DIR / "price_regression_model.joblib"
INSP_MODEL_PATH     = MODELS_DIR / "inspection_failure_model.joblib"
RISK_TABLE_PATH     = MODELS_DIR / "high_risk_model_mileage_pairs.csv"

MANUFACTURER_GROUPS = {
    "Audi", "BMW", "Ford", "Mercedes-Benz", "Opel", "Peugeot",
    "Skoda", "Toyota", "Volkswagen", "Volvo",
}

COUNTRY_MAP = {
    "lithuania": "Lithuania",
    "lt":        "Lithuania",
    "germany":   "Germany",
    "de":        "Germany",
    "sweden":    "Scandinavia",
    "norway":    "Scandinavia",
    "finland":   "Scandinavia",
    "denmark":   "Scandinavia",
    "scandinavia": "Scandinavia",
    "belgium":   "Other_EU",
    "france":    "Other_EU",
    "netherlands": "Other_EU",
    "poland":    "Other_EU",
    "latvia":    "Other_EU",
    "estonia":   "Other_EU",
    "czech":     "Other_EU",
    "austria":   "Other_EU",
    "other eu":  "Other_EU",
    "eu":        "Other_EU",
    "uk":        "Non_EU",
    "usa":       "Non_EU",
    "us":        "Non_EU",
    "japan":     "Non_EU",
    "non eu":    "Non_EU",
    "other":     "Non_EU",
}

VEHICLE_CLASS_MAP = {
    "car":   "M1",
    "suv":   "M1G",
    "4x4":   "M1G",
    "van":   "N1",
    "truck": "N2",
    "bus":   "M2",
    "other": "M1",
}

FUEL_NORM = {
    "petrol":    "BENZINAS",
    "gasoline":  "BENZINAS",
    "benzinas":  "BENZINAS",
    "diesel":    "DYZELINAS",
    "dyzelinas": "DYZELINAS",
    "hybrid":    "BENZINAS / ELEKTRA",
    "electric":  "ELEKTRA",
    "gas":       "DUJOS",
    "lpg":       "DUJOS",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_models() -> tuple:
    if not PRICE_MODEL_PATH.exists():
        sys.exit(f"Price model not found: {PRICE_MODEL_PATH}\n"
                 "Run the combined_som.ipynb notebook first.")
    if not INSP_MODEL_PATH.exists():
        sys.exit(f"Inspection model not found: {INSP_MODEL_PATH}\n"
                 "Run train_inspection_failure_model.py first.")

    price_art = joblib.load(PRICE_MODEL_PATH)
    insp_art  = joblib.load(INSP_MODEL_PATH)
    risk_table = pd.read_csv(RISK_TABLE_PATH) if RISK_TABLE_PATH.exists() else None
    return price_art, insp_art, risk_table


# ── Normalisation helpers ─────────────────────────────────────────────────────

def norm_text(v: object) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "UNKNOWN"
    return re.sub(r"\s+", " ", str(v).strip().upper())


def map_manufacturer(make: str) -> str:
    for grp in MANUFACTURER_GROUPS:
        if grp.lower() in make.lower():
            return grp
    return "Other"


def map_country(raw: str) -> str:
    return COUNTRY_MAP.get(raw.strip().lower(), "Other_EU")


def map_fuel(raw: str) -> str:
    return FUEL_NORM.get(raw.strip().lower(), norm_text(raw))


def map_vehicle_class(raw: str) -> str:
    return VEHICLE_CLASS_MAP.get(raw.strip().lower(), "M1")


# ── Price prediction ──────────────────────────────────────────────────────────

def predict_price(art: dict, row: dict) -> float:
    """Return the smear-adjusted OLS price prediction for a single car."""
    df = pd.DataFrame([row])
    log_pred = art["model"].predict(df)[0]
    return float(np.exp(log_pred) * art["smear"])


# ── Inspection risk ───────────────────────────────────────────────────────────

def lookup_inspection_risk(
    risk_table: pd.DataFrame,
    vehicle_label: str,
    mileage: float,
) -> tuple[float, int, str]:
    """Return (probability, row_count, source) from the pre-computed risk table."""
    label_norm = norm_text(vehicle_label)
    tbl = risk_table[risk_table["vehicle_label"].str.upper().str.strip() == label_norm]

    if tbl.empty:
        return np.nan, 0, "model_not_in_table"

    idx  = (tbl["tp_rida_km"] - mileage).abs().idxmin()
    row  = tbl.loc[idx]
    return float(row["predicted_failure_probability"]), int(row["model_row_count"]), "lookup_nearest_mileage"


def classifier_inspection_risk(
    insp_art: dict,
    vehicle_label: str,
    mileage: float,
    car_age: float,
    fuel: str,
    vehicle_class: str,
    global_failure_rate: float = 0.4736,
    shrinkage_prior: float = 100.0,
) -> float:
    """Fall back to the raw SGD classifier when the lookup table has no entry."""
    hasher     = insp_art["feature_hasher"]
    classifier = insp_art["classifier"]

    row: dict[str, float] = {
        f"model={norm_text(vehicle_label)}": 1.0,
        f"fuel={norm_text(fuel)}": 1.0,
        f"class={norm_text(vehicle_class)}": 1.0,
        "mileage_scaled": mileage / 100_000.0,
    }
    if not np.isnan(car_age):
        row["age_scaled"] = car_age / 40.0
    else:
        row["age_missing"] = 1.0
    row["smokiness_missing"] = 1.0

    x        = hasher.transform([row])
    raw_prob = float(classifier.predict_proba(x)[0, 1])
    # Shrinkage toward global prior — unknown cohort size, use small n=10
    return (10 * raw_prob + shrinkage_prior * global_failure_rate) / (10 + shrinkage_prior)


# ── Risk scoring ──────────────────────────────────────────────────────────────

def price_risk_score(ratio: float) -> float:
    """0-1 score where 1 = maximally suspicious price."""
    if ratio >= 1.0:
        # Overpriced is mildly suspicious (potential hidden quality claim)
        return min(0.25, (ratio - 1.0) * 0.5)
    # Underpriced: score rises sharply below 0.75
    if ratio >= 0.75:
        return (0.75 - ratio) / 0.75 * 0.4          # 0 → 0.4
    return 0.4 + (0.75 - ratio) / 0.75 * 0.6        # 0.4 → 1.0


def combined_risk_score(price_ratio: float, insp_prob: float) -> float:
    """Weighted combination scaled to 0–10."""
    ps = price_risk_score(price_ratio)
    return round((0.4 * ps + 0.6 * insp_prob) * 10, 1)


def risk_label(score: float) -> tuple[str, str]:
    """(label, ANSI colour code)."""
    if score >= 7.5: return "CRITICAL",  "\033[91m"   # bright red
    if score >= 5.5: return "HIGH",      "\033[33m"   # yellow
    if score >= 3.5: return "MODERATE",  "\033[93m"   # light yellow
    return               "LOW",      "\033[92m"   # green

RESET = "\033[0m"
BOLD  = "\033[1m"


# ── CLI input ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-car purchase risk assessment")
    p.add_argument("--make",          type=str,   help="Manufacturer (e.g. Opel)")
    p.add_argument("--model",         type=str,   help="Model name (e.g. Astra)")
    p.add_argument("--year",          type=int,   help="First registration year")
    p.add_argument("--mileage",       type=float, help="Mileage in km")
    p.add_argument("--engine-cc",     type=float, help="Engine displacement in cc")
    p.add_argument("--horsepower",    type=float, help="Horsepower (AG/HP)")
    p.add_argument("--fuel",          type=str,   default="Petrol",
                   help="Fuel type: Petrol / Diesel / Hybrid / Electric / Gas")
    p.add_argument("--country",       type=str,   default="Lithuania",
                   help="Country of first registration")
    p.add_argument("--vehicle-type",  type=str,   default="car",
                   help="Vehicle type: car / suv / 4x4 / van / truck")
    p.add_argument("--has-defects",   action="store_true",
                   help="Listing states the car has defects")
    p.add_argument("--ti-months",     type=float, default=0,
                   help="Months until technical inspection expires (negative if expired)")
    p.add_argument("--asking-price",  type=float, help="Asking price in EUR")
    return p.parse_args()


def _ask(prompt: str, default=None, cast=str):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return cast(raw) if raw else default
        except ValueError:
            print(f"    Please enter a valid {cast.__name__}.")


def prompt_interactively(args: argparse.Namespace) -> argparse.Namespace:
    print(f"\n{BOLD}── Car Purchase Risk Assessment ─────────────────────────────{RESET}")
    print("  Enter car details (press Enter to accept default where shown)\n")
    if args.make          is None: args.make          = _ask("Manufacturer (e.g. Opel, BMW, Toyota)")
    if args.model         is None: args.model         = _ask("Model name (e.g. Astra, 3 Series, Corolla)")
    if args.year          is None: args.year          = _ask("First registration year", cast=int)
    if args.mileage       is None: args.mileage       = _ask("Mileage (km)", cast=float)
    if args.engine_cc     is None: args.engine_cc     = _ask("Engine size (cc, e.g. 1600)", cast=float, default=1800)
    if args.horsepower    is None: args.horsepower    = _ask("Horsepower (e.g. 115)", cast=float, default=110)
    if args.asking_price  is None: args.asking_price  = _ask("Asking price (EUR)", cast=float)

    raw_fuel = _ask("Fuel type (Petrol/Diesel/Hybrid/Electric/Gas)", default=args.fuel)
    args.fuel = raw_fuel

    raw_country = _ask("Country of first registration (e.g. Lithuania, Germany, Other EU)", default=args.country)
    args.country = raw_country

    raw_vtype = _ask("Vehicle type (car/SUV/4x4/van/truck)", default=args.vehicle_type)
    args.vehicle_type = raw_vtype

    raw_defects = _ask("Has listed defects? (y/n)", default="n")
    args.has_defects = raw_defects.lower() in ("y", "yes", "1", "true")

    args.ti_months = _ask("Months until inspection expires (negative = expired, 0 = unknown)",
                          cast=float, default=args.ti_months)
    return args


# ── Report ────────────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 30) -> str:
    filled = round(score / 10 * width)
    return "█" * filled + "░" * (width - filled)


def print_report(
    make: str, model: str, year: int, mileage: float,
    asking_price: float, predicted_price: float,
    insp_prob: float, insp_count: int, insp_source: str,
    score: float,
) -> None:
    label, colour = risk_label(score)
    ratio = asking_price / predicted_price

    price_flag = (
        "SEVERELY UNDERPRICED"  if ratio < 0.55 else
        "NOTABLY UNDERPRICED"   if ratio < 0.75 else
        "SLIGHTLY UNDERPRICED"  if ratio < 0.90 else
        "FAIR MARKET PRICE"     if ratio < 1.10 else
        "ABOVE MARKET PRICE"    if ratio < 1.30 else
        "SIGNIFICANTLY OVERPRICED"
    )

    insp_flag = (
        "NEAR-CERTAIN FAILURE"  if insp_prob > 0.90 else
        "HIGH RISK"             if insp_prob > 0.75 else
        "MODERATE RISK"         if insp_prob > 0.55 else
        "LOWER RISK"
    )

    sep = "═" * 62
    thin = "─" * 62

    print(f"\n{BOLD}{sep}{RESET}")
    print(f"{BOLD}  PURCHASE RISK ASSESSMENT{RESET}")
    print(thin)
    print(f"  Vehicle       {BOLD}{make} {model} ({year}){RESET}")
    print(f"  Mileage       {mileage:,.0f} km")
    print(f"  Asking price  €{asking_price:,.0f}")
    print(thin)

    print(f"\n  {BOLD}Price Analysis{RESET}")
    print(f"  Predicted fair market value  €{predicted_price:,.0f}")
    print(f"  Actual / Predicted ratio     {ratio:.2f}  ({price_flag})")
    if ratio < 1.0:
        diff = predicted_price - asking_price
        print(f"  Discount vs market           €{diff:,.0f}  ({(1-ratio)*100:.0f}% below predicted)")

    print(f"\n  {BOLD}Inspection Failure Risk{RESET}")
    print(f"  Probability of failing next   {insp_prob*100:.1f}%  ({insp_flag})")
    if insp_count > 0:
        print(f"  Based on                      {insp_count:,} inspections of this model")
    print(f"  Source                        {insp_source}")

    print(f"\n  {BOLD}Combined Risk Score{RESET}")
    bar_str = _bar(score)
    print(f"  {colour}{BOLD}▐{bar_str}▌  {label}   {score} / 10{RESET}")

    print(f"\n  {BOLD}Interpretation{RESET}")
    if label == "CRITICAL":
        print("  This listing shows both a significantly below-market price")
        print("  and a high probability of imminent inspection failure.")
        print("  This is the strongest fraud/poor-condition profile.")
        print("  → Independent mechanical inspection strongly recommended.")
        print("  → Request full service history and VIN check before purchase.")
    elif label == "HIGH":
        print("  One or both risk signals are elevated.")
        print("  → Request a pre-purchase inspection by an independent mechanic.")
        print("  → Verify service history and check for prior accident records.")
    elif label == "MODERATE":
        print("  Some risk present but not alarming.")
        print("  → Standard due diligence: verify mileage, check for recalls.")
    else:
        print("  Risk profile appears normal for this type of vehicle.")
        print("  → Standard checks apply: test drive, service history review.")

    print(f"\n{BOLD}{sep}{RESET}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Prompt for any missing required fields
    missing = any(v is None for v in [
        args.make, args.model, args.year, args.mileage, args.asking_price
    ])
    if missing:
        args = prompt_interactively(args)

    # Validate
    if args.year < 1950 or args.year > SCORING_YEAR:
        sys.exit(f"Invalid registration year: {args.year}")
    if args.mileage < 0:
        sys.exit("Mileage cannot be negative.")
    if args.asking_price <= 0:
        sys.exit("Asking price must be positive.")

    price_art, insp_art, risk_table = load_models()

    make  = args.make.strip()
    model = args.model.strip()
    vehicle_label = f"{make} {model}"

    age            = SCORING_YEAR - args.year
    manufacturer_grp = map_manufacturer(make)
    country_grp      = map_country(args.country)
    fuel_norm        = map_fuel(args.fuel)
    vehicle_class    = map_vehicle_class(args.vehicle_type)

    # ── Price prediction ──────────────────────────────────────────────────────
    price_row = {
        "age":             age,
        "mileage_km":      args.mileage,
        "engine_cc":       args.engine_cc or 1800,
        "horsepower_ag":   args.horsepower or 110,
        "ti_months":       args.ti_months,
        "has_defects":     bool(args.has_defects),
        "country_grp":     country_grp,
        "manufacturer_grp": manufacturer_grp,
        "price_eur":       args.asking_price,   # needed for formula namespace
    }
    predicted_price = predict_price(price_art, price_row)

    # ── Inspection risk ───────────────────────────────────────────────────────
    insp_prob, insp_count, insp_source = np.nan, 0, "not_available"

    if risk_table is not None:
        insp_prob, insp_count, insp_source = lookup_inspection_risk(
            risk_table, vehicle_label, args.mileage
        )

    if np.isnan(insp_prob):
        # Fall back to live classifier
        insp_prob = classifier_inspection_risk(
            insp_art, vehicle_label, args.mileage,
            float(age), fuel_norm, vehicle_class,
        )
        insp_source = "classifier_fallback"

    # ── Score & report ────────────────────────────────────────────────────────
    score = combined_risk_score(args.asking_price / predicted_price, insp_prob)
    print_report(
        make, model, args.year, args.mileage,
        args.asking_price, predicted_price,
        insp_prob, insp_count, insp_source,
        score,
    )


if __name__ == "__main__":
    main()
