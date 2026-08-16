"""
train_models.py

Trains and compares 4 models for fraud detection:
- Isolation Forest (unsupervised anomaly detection)
- Logistic Regression (supervised, interpretable baseline)
- Random Forest (supervised, tree-based, handles nonlinearity)
- XGBoost (supervised, gradient boosting, usually strongest on tabular data)

WHY these 4 specifically:
- Isolation Forest: frames fraud as "anomaly" rather than "classification" -
  useful because we have so few labeled fraud examples (5) that a purely
  supervised model risks just memorizing those 5 rows rather than learning
  a generalizable pattern.
- Logistic Regression: simple, coefficients are directly interpretable
  (which matters for a forensics tool - "why was this flagged" needs an answer).
- Random Forest / XGBoost: capture nonlinear interactions between ratios
  that a linear model would miss (e.g., high leverage + declining margin
  together might matter more than either alone).

WHY Leave-One-Out Cross-Validation (LOOCV), not train/test split:
With 72 rows and only 5 positives, a single 80/20 split could put 0 or 1
fraud case in the test set - any resulting metric would be statistically
meaningless (one lucky/unlucky row swings the whole score). LOOCV trains
on 71 rows and tests on the 1 left out, repeated 72 times - every row is
tested exactly once, giving a stable estimate despite the tiny sample.

WHY class_weight='balanced' (Logistic Regression, Random Forest):
With 67 healthy vs 5 fraud (~13:1 ratio), an unweighted model can get
93% "accuracy" by just always predicting "healthy" - useless. Balanced
class weights make misclassifying a fraud row cost ~13x more than
misclassifying a healthy row during training, forcing the model to
actually try to find the minority class.

WHY scale_pos_weight for XGBoost (its equivalent of class_weight):
XGBoost doesn't have class_weight='balanced' - its equivalent parameter
is scale_pos_weight, set here as (count of negative) / (count of positive).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from forensiq.ml.build_dataset import build_modeling_dataset
from forensiq.ml.preprocessing import build_pipeline, FEATURE_COLUMNS,build_preprocessor


def run_loocv_supervised(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Runs Leave-One-Out CV for a supervised model.
    Returns predictions + probabilities collected across all 72 folds,
    plus computed metrics.
    WHY we collect predictions across folds instead of averaging per-fold
    metrics: with 1 test row per fold, per-fold precision/recall is
    undefined (division by zero when there's no positive in that single
    row). Instead we pool all 72 out-of-fold predictions into one set,
    THEN compute metrics once - this is the standard, statistically
    correct way to do LOOCV evaluation for classification.
    """
    loo = LeaveOneOut()
    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)[0]
        proba = pipe.predict_proba(X_test)[0][1]  # probability of class 1 (fraud)

        y_true_all.append(y_test.values[0])
        y_pred_all.append(pred)
        y_proba_all.append(proba)

    return {
        "y_true": np.array(y_true_all),
        "y_pred": np.array(y_pred_all),
        "y_proba": np.array(y_proba_all),
    }


def run_loocv_isolation_forest(X: pd.DataFrame, y: pd.Series, contamination: float) -> dict:
    """
    Isolation Forest is unsupervised - it never sees `y` during training.
    We still use LOOCV structure (train on 71, score the 1 held out) so
    the evaluation is apples-to-apples comparable with the supervised models,
    but the model itself only ever learns from the feature patterns of the
    71 training rows, not their labels.

    contamination = expected proportion of anomalies in training data.
    We set this to match our known fraud rate (5/72 ~= 0.07) so the model's
    internal anomaly threshold roughly aligns with our actual base rate.
    """
    loo = LeaveOneOut()
    y_true_all = []
    y_pred_all = []
    y_score_all = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        pipe = build_pipeline(
            IsolationForest(contamination=contamination, random_state=42)
        )
        pipe.fit(X_train)

        # IsolationForest.predict returns -1 (anomaly) or 1 (normal).
        # We flip this to match our convention: 1 = fraud/anomaly, 0 = normal.
        raw_pred = pipe.predict(X_test)[0]
        pred = 1 if raw_pred == -1 else 0

        # decision_function: lower score = more anomalous. We negate it so
        # higher = more anomalous, matching "higher proba = more fraud-like"
        # convention used by the supervised models' proba output.
        score = -pipe.named_steps["model"].score_samples(
            pipe.named_steps["preprocessor"].transform(X_test)
        )[0]

        y_true_all.append(y_test.values[0])
        y_pred_all.append(pred)
        y_score_all.append(score)

    return {
        "y_true": np.array(y_true_all),
        "y_pred": np.array(y_pred_all),
        "y_proba": np.array(y_score_all),  # anomaly score, not a calibrated probability
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Precision/Recall/F1/ROC-AUC + confusion matrix.
    zero_division=0 handles the edge case where a model predicts zero
    positives at all (precision undefined -> report as 0 rather than crash).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }

from sklearn.metrics import fbeta_score


def threshold_sweep_table(y_true: np.ndarray, y_proba: np.ndarray, thresholds: list) -> pd.DataFrame:
    """
    Builds a table showing Precision/Recall/F1/F2/confusion-matrix at
    multiple probability thresholds, using the SAME LOOCV out-of-fold
    predictions already computed - no retraining needed, we're just
    changing the cutoff applied to probabilities we already have.

    WHY F2 alongside F1:
    F1 weighs Precision and Recall equally. F2 weighs Recall 2x more
    than Precision - matching our stated priority: minimize both false
    negatives and false positives, but lean harder against false
    negatives (missing real fraud) since that's the costlier mistake
    for a forensics decision-support tool.

    WHY sweep multiple thresholds instead of just picking one:
    The default 0.5 cutoff is arbitrary - it's not calibrated to our
    specific cost trade-off (FN costlier than FP). Sweeping shows us
    exactly how Recall/Precision move as we loosen or tighten the
    cutoff, so we pick a threshold backed by evidence, not a guess.
    """
    rows = []
    for t in thresholds:
        y_pred_at_t = (y_proba >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_at_t).ravel()
        precision = precision_score(y_true, y_pred_at_t, zero_division=0)
        recall = recall_score(y_true, y_pred_at_t, zero_division=0)
        f1 = f1_score(y_true, y_pred_at_t, zero_division=0)
        f2 = fbeta_score(y_true, y_pred_at_t, beta=2, zero_division=0)

        rows.append({
            "threshold": t,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "f2": round(f2, 3),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })

    return pd.DataFrame(rows)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def run_loocv_with_smote(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Same as run_loocv_supervised, but applies SMOTE (Synthetic Minority
    Oversampling) to the TRAINING fold only, before fitting the model.

    WHY SMOTE might help here:
    With only 5 fraud examples across 71 training rows per fold, the
    model barely sees enough fraud patterns to learn from. SMOTE creates
    synthetic fraud-like rows by interpolating between existing fraud
    examples' feature values - giving the model more "practice material"
    without needing real new data (which we don't have).

    WHY SMOTE must be INSIDE the pipeline, applied only to training data:
    If we synthesized new rows before splitting into train/test, a
    synthetic row could end up in the test set while being derived from
    training rows - leaking information. Using imblearn's Pipeline
    (not sklearn's) ensures SMOTE only ever touches the training fold.

    CAVEAT: SMOTE needs at least k_neighbors+1 minority samples to work.
    With only 4 fraud rows per training fold (5 total minus 1 in test),
    we use k_neighbors=3 (default is 5, which would fail here).
    """
    loo = LeaveOneOut()
    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = ImbPipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=42, k_neighbors=3)),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)[0]
        proba = pipe.predict_proba(X_test)[0][1]

        y_true_all.append(y_test.values[0])
        y_pred_all.append(pred)
        y_proba_all.append(proba)

    return {
        "y_true": np.array(y_true_all),
        "y_pred": np.array(y_pred_all),
        "y_proba": np.array(y_proba_all),
    }


def main():
    df = build_modeling_dataset()

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    contamination_rate = n_pos / len(y)
    scale_pos_weight = n_neg / n_pos

    print(f"Dataset: {len(df)} rows, {n_pos} fraud, {n_neg} healthy\n")
    print("=" * 60)

    results = {}

    # --- Isolation Forest (unsupervised) ---
    print("\nTraining Isolation Forest (unsupervised)...")
    iso_results = run_loocv_isolation_forest(X, y, contamination=contamination_rate)
    results["Isolation Forest"] = compute_metrics(**iso_results)

    # --- Logistic Regression ---
    print("Training Logistic Regression...")
    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_results = run_loocv_supervised(log_reg, X, y)
    results["Logistic Regression"] = compute_metrics(**lr_results)

    # --- Random Forest ---
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
    rf_results = run_loocv_supervised(rf, X, y)
    results["Random Forest"] = compute_metrics(**rf_results)

    # --- XGBoost ---
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_results = run_loocv_supervised(xgb, X, y)
    results["XGBoost"] = compute_metrics(**xgb_results)

    # --- Threshold sweep for Logistic Regression (our recall-priority candidate) ---
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP — Logistic Regression")
    print("=" * 60)
    thresholds_to_try = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    sweep_df = threshold_sweep_table(
        lr_results["y_true"], lr_results["y_proba"], thresholds_to_try
    )
    print(sweep_df.to_string(index=False))

    # --- Report ---
    print("\n" + "=" * 60)
    print("RESULTS (Leave-One-Out CV, pooled across all 72 folds)")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        print(f"  Precision : {metrics['precision']:.3f}")
        print(f"  Recall    : {metrics['recall']:.3f}")
        print(f"  F1        : {metrics['f1']:.3f}")
        print(f"  ROC-AUC   : {metrics['roc_auc']:.3f}")
        cm = metrics["confusion_matrix"]
        print(f"  Confusion : TN={cm['TN']} FP={cm['FP']} FN={cm['FN']} TP={cm['TP']}")

        # --- Logistic Regression + SMOTE (attempt to improve recall via oversampling) ---
    print("\nTraining Logistic Regression + SMOTE...")
    log_reg_smote = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_smote_results = run_loocv_with_smote(log_reg_smote, X, y)
    results["Logistic Regression + SMOTE"] = compute_metrics(**lr_smote_results)

    print("\nTHRESHOLD SWEEP — Logistic Regression + SMOTE")
    sweep_smote_df = threshold_sweep_table(
        lr_smote_results["y_true"], lr_smote_results["y_proba"], thresholds_to_try
    )
    print(sweep_smote_df.to_string(index=False))


if __name__ == "__main__":
    main()