import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from minisom import MiniSom
np.random.seed(42)

# ── Colours ───────────────────────────────────────────
C_LEGIT = "#2196F3"   # blue  — normalus skelbimas
C_FRAUD = "#F44336"   # red   — įtartinas skelbimas

# ── Sizes ─────────────────────────────────────────────
FIG_WIDE = (14, 5)
FIG_SQUARE = (6, 5)
FIG_TALL = (8, 6)

path = ".../autoplius_cars_03_17.csv"

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 11
print("Style settings ready.")

SOM_PARAMS = {
    "map_size": 18,
    "learning_rate": 0.1,
    "sigma": None,
    "iterations": None,
    # "gaussian", "mexican_hat, "bubble"
    "neighborhood_function": "gaussian",
    "topology": "rectangular",
}

FRAUD_PARAMS = {
    "suspect_pct": 5,
    "top_k": 20,
}

df_raw = pd.read_csv(path)

def clean_and_engineer(df):
    df = df.copy()

    df["price_eur"] = (
        df["price_eur"].astype(str)
        .str.replace("\xa0", "").str.replace(" ", "").str.strip()
    )
    df["price_eur"] = pd.to_numeric(df["price_eur"], errors="coerce")

    df["mileage_km"] = (
        df["mileage_km"].astype(str)
        .str.replace(" km", "").str.replace("\xa0", "").str.replace(" ", "").str.strip()
    )
    df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")

    def parse_engine(s):
        vol, hp = np.nan, np.nan
        if pd.isna(s): return vol, hp
        s = str(s)
        if "cm³" in s:
            try: vol = float(s.split("cm³")[0].replace(" ", "").replace("\xa0", ""))
            except: pass
        if "AG" in s:
            try: hp = float(s.split("AG")[0].split(",")[-1].strip().replace(" ", ""))
            except: pass
        return vol, hp

    parsed = df["engine"].apply(parse_engine)
    df["engine_cc"] = parsed.apply(lambda x: x[0])
    df["engine_hp"] = parsed.apply(lambda x: x[1])

    ref = pd.Timestamp("2026-03-17")
    df["first_reg_dt"] = pd.to_datetime(df["first_registration"], format="%Y-%m", errors="coerce")
    df["car_age_years"] = ((ref - df["first_reg_dt"]).dt.days / 365.25).round(2)
    df["tech_insp_dt"] = pd.to_datetime(df["technical_inspection_until"], format="%Y-%m", errors="coerce")
    df["months_until_insp"] = ((df["tech_insp_dt"] - ref).dt.days / 30.44).round(1)

    df["has_vin_num"] = df["has_vin"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0)
    df["rim_num"] = pd.to_numeric(
        df["rim_diameter"].astype(str).str.replace("R", "").str.strip(), errors="coerce"
    )

    df["km_per_year"] = df["mileage_km"] / (df["car_age_years"].replace(0, np.nan))
    df["price_per_hp"] = df["price_eur"]  / (df["engine_hp"].replace(0, np.nan))
    df["price_per_km"] = df["price_eur"]  / (df["mileage_km"].replace(0, np.nan))
    df["year_mismatch"] = (df["model_year"] - df["first_reg_dt"].dt.year).abs()
    df["price_to_model"] = df.groupby("model")["price_eur"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    df["price_to_segment"]   = df.groupby(["fuel_type", "body_type"])["price_eur"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    df["km_to_age_group"] = df.groupby(pd.cut(df["car_age_years"].fillna(0), bins=[0, 3, 7, 12, 20, 100], labels=False))["mileage_km"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

    return df

df_raw = clean_and_engineer(df_raw)

FEATURES = [
    "price_eur", "mileage_km", "engine_cc", "engine_hp",
    "model_year", "car_age_years", "has_vin_num",
    "price_per_hp", "km_per_year", "price_per_km", "year_mismatch",
    "price_to_model", "km_to_age_group", "price_to_segment",
    "months_until_insp", "curb_weight_kg", "seat_count", "rim_num",
]

# Atsitiktinis split: 80% train, 20% test
def prep_X(df):
    X = df[FEATURES].copy().replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())
    return X.values

X_all = prep_X(df_raw)
df_train, df_test, X_train_raw, X_test_raw = train_test_split(
    df_raw, X_all, test_size=0.2, random_state=42
)
df_train = df_train.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

print(f"Train: {len(X_train):,} ads")
print(f"Test:  {len(X_test):,} ads")
print(df_raw[FEATURES].describe().round(1).to_string())

# ── Overlapping histograms ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIG_WIDE)
fig.suptitle("Pagrindinių požymių pasiskirstymas - Train vs Test", fontweight="bold")

for ax, feat, title in zip(axes,
        ["price_eur", "mileage_km", "car_age_years"],
        ["Kaina (EUR)", "Rida (km)", "Amžius (metai)"]):
    ax.hist(df_train[feat].dropna(), bins=40, alpha=0.7, color=C_LEGIT, label="Train", density=True)
    ax.hist(df_test[feat].dropna(),  bins=40, alpha=0.6, color=C_FRAUD, label="Test",  density=True)
    ax.set_title(title, fontweight="bold")
    ax.legend()

plt.tight_layout()
plt.show()

# ── Požymių koreliacijos ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(pd.DataFrame(X_train_raw, columns=FEATURES).corr(),
            cmap="coolwarm", center=0, linewidths=0.3, annot=False, ax=ax)
ax.set_title("Požymių koreliacijos matrica (train)", fontweight="bold")
plt.tight_layout()
plt.show()

n = X_train.shape[0]
recommended_neurons = 5 * np.sqrt(n)
map_size = SOM_PARAMS["map_size"]
sigma = SOM_PARAMS["sigma"] if SOM_PARAMS["sigma"] is not None else map_size * 0.5
iterations = SOM_PARAMS["iterations"] if SOM_PARAMS["iterations"] is not None else int(recommended_neurons * 500)

print(f"Recommended SOM parameters for {n} samples:")
print(f"  Map size:      {map_size}x{map_size} ({map_size**2} neurons)")
print(f"  Iterations:    {iterations}")
print(f"  Learning rate: {SOM_PARAMS['learning_rate']}")
print(f"  Sigma:         {sigma}")

som = MiniSom(
    x=map_size, y=map_size,
    input_len=X_train.shape[1],
    sigma=sigma,
    learning_rate=SOM_PARAMS["learning_rate"],
    neighborhood_function=SOM_PARAMS["neighborhood_function"],
    topology=SOM_PARAMS["topology"],
    random_seed=42,
)
som.random_weights_init(X_train)
som.train_batch(X_train, num_iteration=iterations, verbose=True)

# SOM požymių ištraukimas
def extract_som_features(som, X, map_size):
    clusters, dists = [], []
    for x in X:
        bmu = som.winner(x)
        bmu_w = som.get_weights()[bmu[0], bmu[1]]
        clusters.append(bmu[0] * 10000 + bmu[1])
        dists.append(np.linalg.norm(x - bmu_w))
    return np.array(clusters, dtype=float), np.array(dists)

som_cluster_train, som_dist_train = extract_som_features(som, X_train, map_size)
som_cluster_test, som_dist_test = extract_som_features(som, X_test, map_size)

print("SOM features ready.")
print(f"  Clusters: {len(np.unique(som_cluster_train))} unique | "f"distance range [{som_dist_train.min():.2f}, {som_dist_train.max():.2f}]")

threshold_dist = np.percentile(som_dist_train, 100 - FRAUD_PARAMS["suspect_pct"])
suspect_train = som_dist_train > threshold_dist
suspect_test = som_dist_test > threshold_dist

# U-Matrix ir atstumo pasiskirstymas
fig, axes = plt.subplots(1, 2, figsize=FIG_WIDE)
fig.suptitle("SOM: Klasteriai", fontweight="bold")

ax = axes[0]
im = ax.imshow(som.distance_map().T, cmap="bone_r", aspect="auto")
plt.colorbar(im, ax=ax, label="Vid. atstumas iki kaimynų")
for x_pt, is_suspect in zip(X_train, suspect_train):
    bmu = som.winner(x_pt)
    ax.plot(bmu[0] + .5, bmu[1] + .5,
            "o", color=C_FRAUD if is_suspect else C_LEGIT,
            alpha=0.3, ms=4)
ax.set_title("U-Matrix - raudona = įtartinas (train)")

ax2 = axes[1]
ax2.hist(som_dist_train[~suspect_train], bins=40, alpha=0.6,
         color=C_LEGIT, label="Normalus", density=True)
ax2.hist(som_dist_train[suspect_train],  bins=40, alpha=0.6,
         color=C_FRAUD, label=f"Įtartinas (top {FRAUD_PARAMS['suspect_pct']}%)", density=True)
ax2.axvline(threshold_dist, color="black", lw=1.5, linestyle="--",
            label=f"Slenkstis ({threshold_dist:.2f})")
ax2.set_title("SOM atstumo požymis\naukštas = neįprastas skelbimas")
ax2.set_xlabel("Atstumas iki BMU")
ax2.legend()

plt.tight_layout()
plt.show()

# Top įtartiniausi (test)
results = df_test[["ad_id", "model", "model_year", "price_eur", "mileage_km"]].copy().reset_index(drop=True)
results["som_cluster"] = som_cluster_test.astype(int)
results["som_distance"] = som_dist_test.round(4)
results["suspect"] = suspect_test

top_k = FRAUD_PARAMS["top_k"]
top_results = results.nlargest(top_k, "som_distance")
print(f"\nTop-{top_k} according to SOM distance (test):")
print(top_results.to_string(index=False))