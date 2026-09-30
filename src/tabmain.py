import logging
import os
from pathlib import Path

# TabPFN's v2.5/v2.6/v3 checkpoints are gated: downloading them requires a
# browser-based license acceptance against api.priorlabs.ai. Only v2 is
# ungated and downloads straight from HuggingFace, so pin it here to keep this
# script runnable offline and non-interactively. This must be set before
# `tabpfn` is imported, since its settings are read from the environment at
# import time. Override with TABPFN_MODEL_VERSION=v3 once you have accepted
# the license.
os.environ.setdefault("TABPFN_MODEL_VERSION", "v2")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tabpfn import TabPFNClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Constants
# Class 0 is not a fluid and not a missing label: an absent FLUID reading means the
# interval is non-sandstone, so there is no reservoir to hold a fluid. Confirmed
# against LITHO in the raw data - FLUID is populated for 99.9% of LITHO==1
# (sandstone) rows and for essentially none of LITHO in (2, 4, 5, 6). Codes 7-12
# are the actual fluids and map to 1-6.
FLUID_MAP = {0: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6}
INVALID_VALUES = {
    'RT': -999.0,
    'RHOB': -999.0,
    'NPHI': -9.99,
}
DEFAULT_MODEL_PATH = Path("models/model_tabpfn.pkl")
DEFAULT_PLOT_PATH = Path("reports/tabpfn_evaluation.png")

# TabPFN is an in-context learner: the whole training set is fed through the
# transformer as context at inference time, and attention cost is quadratic in
# that context. The pretraining regime it was built for tops out here
# (tabpfn.inference_config.InferenceConfig.MAX_NUMBER_OF_SAMPLES), and going
# past it degrades accuracy long before it exhausts GPU memory.
TABPFN_MAX_TRAIN_SAMPLES = 10_000

# Columns excluded from IQR outlier filtering. DEPTH is a survey coordinate, not
# a measurement, so its tails are the top and bottom of the wells rather than
# bad readings.
OUTLIER_EXEMPT_COLUMNS = ('DEPTH',)


def _select_tabpfn_device(requested_device: str | None = None) -> str:
    """Selects a safe device for TabPFN, preferring CUDA only when available."""
    if requested_device:
        if requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning("Requested CUDA for TabPFN but CUDA is not available; falling back to CPU.")
            return "cpu"
        return requested_device

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_and_preprocess_data(
    path: str | Path,
    test_size: float = 0.2,
    random_state: int = 0,
    drop_non_reservoir: bool = True,
):
    """Loads well-log CSV data, filters sentinel null values, encodes FLUID target,
    applies log-transformation to RT, and splits into train/test sets.

    An absent FLUID reading means the interval is non-sandstone (class 0). Lithology
    is interpreted separately upstream, so this model is only ever asked to score
    intervals already known to be sandstone - the default therefore drops class 0
    and trains a pure fluid discriminator on classes 1-6. Spending no context on
    non-reservoir rock is what makes the rare fluids learnable at all.

    Set `drop_non_reservoir=False` to keep class 0 and predict reservoir-vs-fluid in
    one stage, for scoring raw logs with no lithology interpretation available."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)

    # 1. Resolve absent FLUID readings, then map categories safely
    if drop_non_reservoir:
        non_reservoir = int(df['FLUID'].isna().sum())
        logger.info(
            f"Dropping {non_reservoir} of {len(df)} non-sandstone rows "
            f"({non_reservoir / len(df):.1%} of the file); keeping fluid classes only"
        )
        df = df[df['FLUID'].notna()]
        fluid_codes = df['FLUID']
    else:
        fluid_codes = df['FLUID'].fillna(0)

    unmapped = set(fluid_codes.unique()) - set(FLUID_MAP)
    if unmapped:
        raise ValueError(
            f"Unrecognized FLUID codes {sorted(unmapped)} in {path}; "
            f"expected one of {sorted(FLUID_MAP)}"
        )
    df = df.assign(FLUID=fluid_codes.map(FLUID_MAP).astype(int))

    # 2. Vectorized filtering of null/sentinel values (-999, -9.99)
    valid_mask = pd.Series(True, index=df.index)
    for col, sentinel in INVALID_VALUES.items():
        if col in df.columns:
            valid_mask &= df[col] != sentinel

    # RT must be strictly positive: log(0) is -inf and log(<0) is NaN, and -inf
    # survives the dropna() in clean_data and would reach the model.
    valid_mask &= df['RT'] > 0

    dropped = len(df) - int(valid_mask.sum())
    logger.info(f"Dropped {dropped} of {len(df)} rows with sentinel or non-positive values")
    df = df[valid_mask].reset_index(drop=True)

    # 3. Log-transform resistivity and drop original RT column
    df['LOG_RT'] = np.log(df['RT'])
    df = df.drop(columns=['RT'])

    # 4. Feature / Target Split
    X = df.drop(columns=['FLUID'])
    y = df['FLUID']

    # 5. Train/Test Split (stratified to preserve fluid class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, y_train, X_test, y_test


def clean_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    drop_columns: tuple[str, ...] = ('LITHO',),
    ):
    """Cleans well-log train/test features by dropping unneeded columns, removing
    physically invalid values, and filtering training outliers using the Interquartile
    Range (IQR). Missing values are preserved - TabPFN handles NaN natively.

    `drop_columns` is the set of non-predictive columns to remove. Add 'DEPTH' to
    test whether the model is keying on well position rather than fluid physics -
    depth is a survey coordinate, so any signal it carries is unlikely to transfer
    to a well outside the training set's depth ranges."""
    # 1. Drop unnecessary columns safely
    X_train = X_train.drop(columns=list(drop_columns), errors='ignore')
    X_test = X_test.drop(columns=list(drop_columns), errors='ignore')

    # 2. Drop physically invalid readings on Training Data. Negated comparisons keep
    #    NaN rows: `NaN > 0` is False, so the positive form would discard them too.
    train_mask = ~(X_train['RHOB'] <= 0) & ~(X_train['NPHI'] <= 0)
    X_train = X_train[train_mask]

    # 3. Apply IQR outlier filtering ONLY on Training Data, and only on the
    #    measured logs - see OUTLIER_EXEMPT_COLUMNS.
    log_columns = X_train.columns.difference(OUTLIER_EXEMPT_COLUMNS)
    logs = X_train[log_columns]

    Q1 = logs.quantile(0.25)
    Q3 = logs.quantile(0.75)
    IQR = Q3 - Q1

    is_outlier = ((logs < (Q1 - 1.5 * IQR)) | (logs > (Q3 + 1.5 * IQR))).any(axis=1)
    X_train = X_train[~is_outlier]

    # Align y_train with the finalized X_train index
    y_train = y_train.loc[X_train.index]

    # 4. Test Set is left as-is - no outlier filtering and no NaN removal, so the
    #    evaluation reflects every interval the model will see in production.
    y_test = y_test.loc[X_test.index]

    logger.info(f"Cleaned X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"Cleaned X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    return X_train, y_train, X_test, y_test


def _stratified_subsample(
    X: pd.DataFrame, y: pd.Series, n_samples: int | None, random_state: int, label: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Draws a class-proportional subsample, or returns the input untouched if it
    already fits. Logs the reduction so a capped run is never mistaken for a full one."""
    if n_samples is None or len(X) <= n_samples:
        return X, y

    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n_samples, random_state=random_state, stratify=y
    )
    logger.info(
        f"Subsampled {label} from {len(X)} to {len(X_sub)} rows (stratified). "
        f"Class counts: {y_sub.value_counts().sort_index().to_dict()}"
    )
    return X_sub, y_sub


def train_tabpfn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    plot_path: Path | str = DEFAULT_PLOT_PATH,
    max_train_samples: int | None = TABPFN_MAX_TRAIN_SAMPLES,
    max_eval_samples: int | None = 50_000,
    n_importance_samples: int = 1_000,
    random_state: int = 0,
    **hyperparams,
) -> tuple[TabPFNClassifier, dict[str, float]]:
    """Trains a TabPFNClassifier, evaluates performance, saves the confusion matrix
    and permutation feature importances side-by-side as a PNG, and saves the model
    artifact.

    The training set is subsampled to `max_train_samples` because TabPFN carries it
    as transformer context rather than fitting weights to it. Evaluation is capped at
    `max_eval_samples` purely for runtime; pass None for the full test set."""
    # 1. Fit TabPFN within the context budget it was pretrained for
    X_fit, y_fit = _stratified_subsample(
        X_train, y_train, max_train_samples, random_state, "training set"
    )
    X_eval, y_eval = _stratified_subsample(
        X_test, y_test, max_eval_samples, random_state, "test set"
    )

    requested_device = hyperparams.pop("device", None)
    selected_device = _select_tabpfn_device(requested_device)

    default_params = {
        "n_estimators": 8,
        "device": selected_device,
        "random_state": random_state,
    }
    params = {**default_params, **hyperparams}

    logger.info(
        f"Training TabPFNClassifier on device={params['device']} "
        f"with {len(X_fit)} context rows, {len(X_eval)} eval rows..."
    )

    def fit_and_predict(model_params: dict):
        model = TabPFNClassifier(**model_params)
        model.fit(X_fit, y_fit)
        # TabPFN does its heavy lifting in predict, not fit, so the prediction pass
        # has to sit inside the same try block as the fit for the fallback to catch
        # an out-of-memory failure.
        return model, model.predict(X_eval)

    try:
        model, y_pred = fit_and_predict(params)
    except RuntimeError as exc:
        if params["device"] != "cuda":
            raise

        # CPU inference is capped at MAX_CPU_SAMPLES (1000) by default, well below
        # our context size, so the retry has to lift that cap explicitly.
        logger.warning("TabPFN failed on CUDA (%s). Retrying on CPU - this is slow.", exc)
        cpu_params = {**params, "device": "cpu", "ignore_pretraining_limits": True}
        model, y_pred = fit_and_predict(cpu_params)

    # 2. Metrics Computation
    metrics = {
        "Accuracy": accuracy_score(y_eval, y_pred),
        "Precision": precision_score(y_eval, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_eval, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_eval, y_pred, average="weighted", zero_division=0),
    }

    print("\n" + "=" * 40)
    print(" MODEL PERFORMANCE ")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"{name:<12}: {value:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_eval, y_pred, zero_division=0))

    # 3. Model Persistence
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info(f"Model saved successfully to {save_path.resolve()}")

    # 4. Feature Importance & Confusion Matrix Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_eval, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
        xticklabels=model.classes_, yticklabels=model.classes_,
    )
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_title("Confusion Matrix")

    # Plot Feature Importance. TabPFN exposes no native importances, so this is
    # permutation importance, measured on a small slice to keep the repeated
    # prediction passes affordable.
    X_imp, y_imp = _stratified_subsample(
        X_eval, y_eval, n_importance_samples, random_state, "importance sample"
    )
    logger.info(f"Computing permutation importance over {len(X_imp)} rows...")
    perm = permutation_importance(
        model, X_imp, y_imp, n_repeats=3, random_state=random_state, scoring="accuracy"
    )
    importance_df = pd.DataFrame(
        {"Feature": X_imp.columns, "Importance": perm.importances_mean}
    ).sort_values(by="Importance", ascending=False)

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        palette="viridis",
        ax=axes[1],
        hue="Feature",
        legend=False,
    )
    axes[1].set_title("TabPFN Permutation Importance")
    axes[1].set_xlabel("Mean Accuracy Drop")

    plt.tight_layout()

    figure_path = Path(plot_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Evaluation plots saved to {figure_path.resolve()}")

    return model, metrics


def main():
    logging.basicConfig(level=logging.INFO)

    data_path = Path("/home/rian/python_project/myvenv/fl-analysis/data/raw/combined_new.csv")
    
    # 1. Load Data
    X_train, y_train, X_test, y_test = load_and_preprocess_data(data_path)
    print("Data loaded successfully.")

    # 2. Clean Data
    X_train, y_train, X_test, y_test = clean_data(X_train, y_train, X_test, y_test)
    print("Data cleaned successfully.")

    # 3. Train Model
    model, metrics = train_tabpfn(X_train, y_train, X_test, y_test)
    logger.info(f"Training complete: {metrics}")
    return model, metrics


if __name__ == "__main__":
    main()