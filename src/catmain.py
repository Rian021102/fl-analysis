import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
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
FLUID_MAP = {0: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6}
INVALID_VALUES = {
    'RT': -999.0,
    'RHOB': -999.0,
    'NPHI': -9.99,
}
DEFAULT_MODEL_PATH = Path("models/model_cat.pkl")
DEFAULT_PLOT_PATH = Path("reports/model_evaluation.png")


def load_and_preprocess_data(path: str | Path, test_size: float = 0.2, random_state: int = 0):
    """Loads well-log CSV data, filters sentinel null values, encodes FLUID target, 
    applies log-transformation to RT, and splits into train/test sets."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)

    # 1. Fill missing FLUID values and map categories safely
    fluid_codes = df['FLUID'].fillna(0)
    unmapped = set(fluid_codes.unique()) - set(FLUID_MAP)
    if unmapped:
        raise ValueError(
            f"Unrecognized FLUID codes {sorted(unmapped)} in {path}; "
            f"expected one of {sorted(FLUID_MAP)}"
        )
    df['FLUID'] = fluid_codes.map(FLUID_MAP).astype(int)

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
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
    ):
    """Cleans well-log train/test features by dropping unneeded columns, removing
    physically invalid values, and filtering training outliers using the Interquartile
    Range (IQR). Missing values are preserved - CatBoost handles NaN natively."""
    # 1. Drop unnecessary columns safely
    X_train = X_train.drop(columns=['LITHO'], errors='ignore')
    X_test = X_test.drop(columns=['LITHO'], errors='ignore')

    # 2. Drop physically invalid readings on Training Data. Negated comparisons keep
    #    NaN rows: `NaN > 0` is False, so the positive form would discard them too.
    train_mask = ~(X_train['RHOB'] <= 0) & ~(X_train['NPHI'] <= 0)
    X_train = X_train[train_mask]

    # 3. Apply IQR outlier filtering ONLY on Training Data
    Q1 = X_train.quantile(0.25)
    Q3 = X_train.quantile(0.75)
    IQR = Q3 - Q1

    is_outlier = ((X_train < (Q1 - 1.5 * IQR)) | (X_train > (Q3 + 1.5 * IQR))).any(
        axis=1
    )
    X_train = X_train[~is_outlier]

    # Align y_train with the finalized X_train index
    y_train = y_train.loc[X_train.index]

    # 4. Test Set is left as-is - no outlier filtering and no NaN removal, so the
    #    evaluation reflects every interval the model will see in production.
    y_test = y_test.loc[X_test.index]

    logger.info(f"Cleaned X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"Cleaned X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    return X_train, y_train, X_test, y_test


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    plot_path: Path | str = DEFAULT_PLOT_PATH,
    **hyperparams,
) -> CatBoostClassifier:
    """Trains a CatBoostClassifier, evaluates performance, saves the confusion matrix
    and feature importances side-by-side as a PNG, and saves the model artifact."""
    # 1. Default model parameters
    default_params = {
        "iterations": 1000,
        "learning_rate": 0.1,
        "depth": 6,
        "loss_function": "MultiClass",
        "verbose": False,
    }
    params = {**default_params, **hyperparams}

    logger.info("Training CatBoostClassifier...")
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

    # 2. Metrics Computation
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    print("\n" + "=" * 40)
    print(" MODEL PERFORMANCE ")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"{name:<12}: {value:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 3. Model Persistence
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info(f"Model saved successfully to {save_path.resolve()}")

    # 4. Feature Importance & Confusion Matrix Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_title("Confusion Matrix")

    # Plot Feature Importance
    importance_df = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": model.get_feature_importance(),
        }
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
    axes[1].set_title("CatBoost Feature Importance")
    axes[1].set_xlabel("Importance Score")

    plt.tight_layout()

    figure_path = Path(plot_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Evaluation plots saved to {figure_path.resolve()}")

    return model


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
    model = train_catboost(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()