from __future__ import annotations
import sys, joblib, logging, numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from joblib import Parallel, delayed

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

DATA_PATH     = Path("data/marketing_campaign.csv")
ARTIFACT_ROOT = Path("app/artifacts")
FOLDS         = 5
K_MIN, K_MAX  = 2, 10
PCA_VARIANCE  = 0.95
RANDOM_STATE  = 42
N_JOBS        = 1 

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
OUT_DIR = ARTIFACT_ROOT / STAMP
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    sys.exit(f" CSV not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=";")
log.info("Loaded %s rows × %s columns", len(df), df.shape[1])

df["Age"] = 2025 - df["Year_Birth"]
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"],
                                   format="mixed",
                                   dayfirst=True,
                                   errors="coerce")
df["TenureDays"] = (pd.Timestamp("2025-06-30") - df["Dt_Customer"]).dt.days
df.drop(columns=["ID", "Year_Birth", "Dt_Customer"], inplace=True)

cat_cols = df.select_dtypes("object").columns.tolist()
num_cols = df.select_dtypes(["int64", "float64"]).columns.tolist()
log.info("Numeric %d · Categorical %d", len(num_cols), len(cat_cols))
log.info("Missing values before impute: %d", df.isna().sum().sum())

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale",   RobustScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore"))
])
pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])
pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)

def evaluate_fold(train_idx, test_idx, k: int):
    X_train = pre.fit_transform(df.iloc[train_idx])
    X_train = pca.fit_transform(X_train)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                n_init="auto").fit(X_train)

    X_test = pre.transform(df.iloc[test_idx])
    X_test = pca.transform(X_test)
    labels = km.predict(X_test)

    sil = silhouette_score(X_test, labels)
    ch  = calinski_harabasz_score(X_test, labels)
    db  = davies_bouldin_score(X_test, labels)
    return sil, ch, db

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

results: list[tuple[int, float, float, float, float]] = []

for k in range(K_MIN, K_MAX + 1):
    fold_metrics = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_fold)(train, test, k) for train, test in kf.split(df))
    sil, ch, db = np.mean(fold_metrics, axis=0)
    composite = (sil / 0.5) + (ch / 1e4) + (1 / db)
    results.append((k, sil, ch, db, composite))
    log.info("K=%d  sil=%.3f  ch=%.0f  db=%.3f  comp=%.3f",
             k, sil, ch, db, composite)

best_k, sil_best, ch_best, db_best, best_comp = max(results, key=lambda r: r[4])
log.warning("Selected K=%d (composite=%.3f  silhouette=%.3f)",
            best_k, best_comp, sil_best)

X_full = pre.fit_transform(df)
X_full = pca.fit_transform(X_full)
km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init="auto").fit(X_full)

pipeline = Pipeline([("pre", pre), ("pca", pca), ("km", km)]).fit(df)

joblib.dump(pipeline, OUT_DIR / "kmeans_pipeline.joblib")
joblib.dump({"k": best_k,
             "metrics": results,
             "num_cols": num_cols,
             "cat_cols": cat_cols},
            OUT_DIR / "meta.joblib")

df["cluster"] = km.labels_
(df.groupby("cluster")[num_cols]
   .agg(["count", "mean", "median", "std"])
   .to_csv(OUT_DIR / "cluster_summary.csv"))

print(f"Model and diagnostics saved to {OUT_DIR.relative_to(Path.cwd())}")
