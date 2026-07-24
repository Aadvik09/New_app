"""Core synthesis, validation, and evaluation utilities for BootstrapMD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RNG = np.random.default_rng(42)
VALID_TYPES = ("continuous", "integer", "categorical")
MAX_MODEL_ROWS = 4_000
MAX_SCORECARD_ROWS = 1_500


@dataclass
class Evaluation:
    method: str
    fidelity: float
    utility: float | None
    privacy: float
    grade: str
    details: dict[str, float | str]


def infer_schema(data: pd.DataFrame) -> pd.DataFrame:
    """Infer a human-reviewable schema; identifiers are intentionally categorical."""
    rows = []
    for col in data.columns:
        series = data[col]
        non_null = series.dropna()
        low_cardinality = non_null.nunique(dropna=True) <= min(20, max(5, len(data) * 0.05))
        if pd.api.types.is_numeric_dtype(series) and not low_cardinality:
            kind = "integer" if pd.api.types.is_integer_dtype(series) else "continuous"
        elif pd.api.types.is_numeric_dtype(series) and non_null.nunique() > 20:
            kind = "continuous"
        else:
            kind = "categorical"
        rows.append(
            {
                "column": col,
                "type": kind,
                "missing (%)": round(float(series.isna().mean() * 100), 1),
                "unique": int(series.nunique(dropna=True)),
                "include": True,
            }
        )
    return pd.DataFrame(rows)


def schema_types(schema: pd.DataFrame) -> dict[str, str]:
    included = schema[schema["include"].fillna(True)]
    types = dict(zip(included["column"], included["type"]))
    invalid = {k: v for k, v in types.items() if v not in VALID_TYPES}
    if invalid:
        raise ValueError("Each included column must be continuous, integer, or categorical.")
    return types


def validate_data(data: pd.DataFrame, types: dict[str, str]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for col, kind in types.items():
        values = data[col]
        if values.isna().all():
            issues.append({"severity": "error", "column": col, "finding": "All values are missing."})
        if kind in ("continuous", "integer"):
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.notna().mean() < 0.95:
                issues.append({"severity": "warning", "column": col, "finding": "Some values are not numeric and will be imputed."})
            if kind == "integer" and numeric.dropna().mod(1).abs().gt(1e-9).any():
                issues.append({"severity": "warning", "column": col, "finding": "Non-integer values will be rounded."})
        elif values.astype("string").str.strip().eq("").any():
            issues.append({"severity": "warning", "column": col, "finding": "Blank categories will be treated as missing."})
        if values.nunique(dropna=True) == len(values):
            issues.append({"severity": "info", "column": col, "finding": "All values are unique; this may be an identifier and elevate disclosure risk."})
    return issues


def _clean(data: pd.DataFrame, types: dict[str, str]) -> pd.DataFrame:
    out = data[list(types)].copy()
    for col, kind in types.items():
        if kind in ("continuous", "integer"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(out[col].median())
        else:
            out[col] = out[col].astype("string").replace("", pd.NA).fillna("Missing").astype(str)
    return out


def _finish(data: pd.DataFrame, types: dict[str, str]) -> pd.DataFrame:
    out = data.copy()
    for col, kind in types.items():
        if kind == "integer":
            out[col] = pd.to_numeric(out[col], errors="coerce").round().astype("Int64")
        elif kind == "continuous":
            out[col] = pd.to_numeric(out[col], errors="coerce").round(5)
        else:
            out[col] = out[col].astype(str)
    return out


def smoothed_bootstrap(data: pd.DataFrame, types: dict[str, str], size: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    source = _clean(data, types)
    result = source.iloc[rng.integers(0, len(source), size=size)].reset_index(drop=True).copy()
    for col, kind in types.items():
        if kind in ("continuous", "integer"):
            values = source[col].to_numpy(dtype=float)
            spread = np.nanstd(values, ddof=1)
            bandwidth = 0.0 if not np.isfinite(spread) else 0.08 * spread * len(values) ** (-0.2)
            if bandwidth:
                result[col] = result[col].to_numpy(dtype=float) + rng.normal(0, bandwidth, size)
            result[col] = result[col].clip(np.nanmin(values), np.nanmax(values))
    return _finish(result, types)


def smote_nc(data: pd.DataFrame, types: dict[str, str], size: int, target: str | None, random_state: int = 42) -> pd.DataFrame:
    """A dependency-light SMOTE-NC implementation for mixed-type tabular data."""
    if not target or target not in types or types[target] == "continuous":
        return smoothed_bootstrap(data, types, size, random_state)
    rng = np.random.default_rng(random_state)
    source = _model_sample(_clean(data, types), MAX_MODEL_ROWS, random_state)
    counts = source[target].value_counts()
    weights = pd.Series(1 / counts, index=counts.index)
    weights = weights / weights.sum()
    classes = rng.choice(weights.index.to_numpy(), size=size, p=weights.to_numpy())
    numeric_cols = [c for c, t in types.items() if t in ("continuous", "integer") and c != target]
    categorical_cols = [c for c, t in types.items() if t == "categorical" and c != target]
    chunks = []
    for label in counts.index:
        n = int((classes == label).sum())
        pool = source[source[target] == label].reset_index(drop=True)
        if n == 0:
            continue
        if len(pool) < 2 or not numeric_cols:
            chunks.append(pool.iloc[rng.integers(0, len(pool), size=n)].copy())
            continue
        X = pool[numeric_cols].to_numpy(float)
        scale = np.nanstd(X, axis=0)
        scale[scale == 0] = 1
        neighbors = NearestNeighbors(n_neighbors=min(5, len(pool))).fit(X / scale)
        base_idx = rng.integers(0, len(pool), size=n)
        neighbor_ids = neighbors.kneighbors((X / scale)[base_idx], return_distance=False)
        mate_idx = np.array([rng.choice(ids[ids != base] if np.any(ids != base) else ids) for base, ids in zip(base_idx, neighbor_ids)])
        alpha = rng.random(n)[:, None]
        generated = pool.iloc[base_idx].reset_index(drop=True).copy()
        generated[numeric_cols] = X[base_idx] + alpha * (X[mate_idx] - X[base_idx])
        for col in categorical_cols:
            choose_mate = rng.random(n) < 0.5
            generated.loc[choose_mate, col] = pool.loc[mate_idx[choose_mate], col].to_numpy()
        chunks.append(generated)
    result = pd.concat(chunks, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return _finish(result.iloc[:size], types)


def gaussian_copula(data: pd.DataFrame, types: dict[str, str], size: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    source = _model_sample(_clean(data, types), MAX_MODEL_ROWS, random_state)
    encoded = pd.DataFrame(index=source.index)
    encoders: dict[str, LabelEncoder] = {}
    for col, kind in types.items():
        if kind == "categorical":
            enc = LabelEncoder().fit(source[col].astype(str))
            encoders[col] = enc
            encoded[col] = enc.transform(source[col].astype(str))
        else:
            encoded[col] = source[col].astype(float)
    n = len(encoded)
    latent = np.column_stack([norm.ppf(np.clip((rankdata(encoded[c]) - 0.5) / n, 1e-5, 1 - 1e-5)) for c in encoded])
    covariance = np.corrcoef(latent, rowvar=False)
    covariance = np.atleast_2d(covariance) + np.eye(latent.shape[1]) * 1e-6
    sampled = rng.multivariate_normal(np.zeros(latent.shape[1]), covariance, size=size)
    out = pd.DataFrame(index=range(size))
    for index, col in enumerate(encoded.columns):
        quantiles = norm.cdf(sampled[:, index])
        values = np.quantile(encoded[col], quantiles, method="linear")
        if types[col] == "categorical":
            codes = np.clip(np.rint(values).astype(int), 0, len(encoders[col].classes_) - 1)
            out[col] = encoders[col].inverse_transform(codes)
        else:
            out[col] = values
    return _finish(out, types)


def cart_sequential(data: pd.DataFrame, types: dict[str, str], size: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    source = _model_sample(_clean(data, types), MAX_MODEL_ROWS, random_state)
    columns = list(types)
    generated = pd.DataFrame(index=range(size))
    for index, col in enumerate(columns):
        kind = types[col]
        if index == 0:
            generated[col] = source[col].iloc[rng.integers(0, len(source), size)].to_numpy()
            continue
        context = columns[:index]
        X_train, X_new = _feature_matrix(source[context], generated[context], {c: types[c] for c in context})
        y = source[col]
        if kind == "categorical":
            model = DecisionTreeClassifier(min_samples_leaf=max(2, len(source) // 30), random_state=random_state)
            model.fit(X_train, y.astype(str))
            predicted = model.predict(X_new)
            residual_pool = source[col].astype(str).to_numpy()
            replace = rng.random(size) < 0.12
            predicted[replace] = rng.choice(residual_pool, replace.sum())
            generated[col] = predicted
        else:
            model = DecisionTreeRegressor(min_samples_leaf=max(3, len(source) // 25), random_state=random_state)
            model.fit(X_train, y.astype(float))
            train_prediction = model.predict(X_train)
            residuals = y.to_numpy(float) - train_prediction
            generated[col] = model.predict(X_new) + rng.choice(residuals, size=size)
    return _finish(generated, types)


def _feature_matrix(train: pd.DataFrame, new: pd.DataFrame, types: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    for col, kind in types.items():
        if kind == "categorical":
            encoder = LabelEncoder().fit(pd.concat([train[col].astype(str), new[col].astype(str)]))
            left.append(encoder.transform(train[col].astype(str)))
            right.append(encoder.transform(new[col].astype(str)))
        else:
            left.append(pd.to_numeric(train[col], errors="coerce").fillna(0).to_numpy())
            right.append(pd.to_numeric(new[col], errors="coerce").fillna(0).to_numpy())
    return np.column_stack(left), np.column_stack(right)


def _model_sample(data: pd.DataFrame, limit: int, random_state: int = 42) -> pd.DataFrame:
    """Keep modeling predictable on large clinical exports without changing output size."""
    if len(data) <= limit:
        return data.reset_index(drop=True)
    return data.sample(n=limit, random_state=random_state).reset_index(drop=True)


def evaluate(real: pd.DataFrame, synthetic: pd.DataFrame, types: dict[str, str], target: str | None, method: str) -> Evaluation:
    # Scorecards are comparative diagnostics: bounded stratified-sized samples keep
    # the UI responsive even when the uploaded CSV contains hundreds of thousands of rows.
    original = _model_sample(_clean(real, types), MAX_SCORECARD_ROWS)
    fake = _model_sample(_clean(synthetic, types), MAX_SCORECARD_ROWS)
    numeric = [c for c, t in types.items() if t in ("continuous", "integer")]
    categorical = [c for c, t in types.items() if t == "categorical"]
    ks = float(np.mean([ks_2samp(original[c], fake[c]).statistic for c in numeric])) if numeric else 0.0
    tvds = []
    for col in categorical:
        levels = original[col].astype(str).value_counts(normalize=True).index.union(fake[col].astype(str).value_counts(normalize=True).index)
        left = original[col].astype(str).value_counts(normalize=True).reindex(levels, fill_value=0)
        right = fake[col].astype(str).value_counts(normalize=True).reindex(levels, fill_value=0)
        tvds.append(float(0.5 * np.abs(left - right).sum()))
    tvd = float(np.mean(tvds)) if tvds else 0.0
    corr_delta = 0.0
    if len(numeric) > 1:
        corr_delta = float(np.abs(original[numeric].corr().fillna(0) - fake[numeric].corr().fillna(0)).to_numpy().mean())
    pmse = _propensity_mse(original, fake, types)
    fidelity = float(np.clip(100 * (1 - (0.36 * ks + 0.30 * tvd + 0.20 * corr_delta + 0.14 * pmse)), 0, 100))
    utility = _utility(original, fake, types, target)
    duplicate_rate, dcr, mia_risk = _privacy(original, fake, types)
    privacy = float(np.clip(100 * (1 - (0.55 * duplicate_rate + 0.45 * mia_risk)), 0, 100))
    combined = 0.52 * fidelity + 0.28 * (utility if utility is not None else fidelity) + 0.20 * privacy
    grade = "A" if combined >= 84 else "B" if combined >= 70 else "C" if combined >= 55 else "D"
    details = {"KS": round(ks, 3), "TVD": round(tvd, 3), "corr Δ": round(corr_delta, 3), "pMSE": round(pmse, 3), "duplicates (%)": round(100 * duplicate_rate, 1), "DCR": round(dcr, 3), "MIA risk": round(mia_risk, 3)}
    return Evaluation(method, round(fidelity, 1), None if utility is None else round(utility, 1), round(privacy, 1), grade, details)


def _propensity_mse(real: pd.DataFrame, fake: pd.DataFrame, types: dict[str, str]) -> float:
    labels = np.r_[np.zeros(len(real)), np.ones(len(fake))]
    joint = pd.concat([real, fake], ignore_index=True)
    X, _ = _feature_matrix(joint, joint, types)
    model = LogisticRegression(max_iter=250, solver="lbfgs", random_state=42)
    model.fit(X, labels)
    return float(np.mean((model.predict_proba(X)[:, 1] - 0.5) ** 2) * 4)


def _utility(real: pd.DataFrame, fake: pd.DataFrame, types: dict[str, str], target: str | None) -> float | None:
    if not target or target not in types or types[target] == "continuous" or real[target].nunique() < 2:
        return None
    features = {c: t for c, t in types.items() if c != target}
    if not features:
        return None
    X_real, X_fake = _feature_matrix(real[list(features)], fake[list(features)], features)
    y_real, y_fake = real[target].astype(str), fake[target].astype(str)
    model_tstr = DecisionTreeClassifier(max_depth=8, min_samples_leaf=4, random_state=42).fit(X_fake, y_fake)
    model_trtr = DecisionTreeClassifier(max_depth=8, min_samples_leaf=4, random_state=42).fit(X_real, y_real)
    baseline = accuracy_score(y_real, model_trtr.predict(X_real))
    tstr = accuracy_score(y_real, model_tstr.predict(X_real))
    return float(np.clip(100 * tstr / max(baseline, 1e-6), 0, 100))


def _privacy(real: pd.DataFrame, fake: pd.DataFrame, types: dict[str, str]) -> tuple[float, float, float]:
    real_key = real.astype(str).agg("¦".join, axis=1)
    fake_key = fake.astype(str).agg("¦".join, axis=1)
    duplicates = float(fake_key.isin(set(real_key)).mean())
    X_real, X_fake = _feature_matrix(real, fake, types)
    scale = np.std(X_real, axis=0)
    scale[scale == 0] = 1
    neighbors = NearestNeighbors(n_neighbors=1).fit(X_real / scale)
    distances = neighbors.kneighbors(X_fake / scale)[0].ravel()
    dcr = float(np.median(distances))
    mia_risk = float(np.mean(distances <= np.quantile(distances, 0.1)))
    return duplicates, dcr, mia_risk


def recommend(results: Iterable[Evaluation], goal: str) -> Evaluation:
    candidates = list(results)
    if not candidates:
        raise ValueError("Run at least one synthesis method before requesting a recommendation.")
    if goal == "Maximize privacy":
        return max(candidates, key=lambda item: item.privacy)
    if goal == "Maximize utility":
        return max(candidates, key=lambda item: item.utility if item.utility is not None else -1)
    return max(candidates, key=lambda item: 0.60 * item.fidelity + 0.25 * (item.utility or item.fidelity) + 0.15 * item.privacy)
