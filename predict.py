import lasio
import pandas as pd
import joblib
from pathlib import Path

# Constants
BASE_DIR = Path('P:/project/pythonpro/myvenv/fl-analysis')
MODEL_PATH = BASE_DIR / 'models/model_cat.pkl'
FEATURE_COLUMNS = ['DEPTH', 'GR', 'RT', 'RHOB', 'NPHI']
LABELS = ['Non-SST', 'Gas', 'PosGas', 'Oil', 'PosOil', 'WTR', 'WtrRise']


def load_model(model_path: Path):
    """Load the trained model from disk."""
    return joblib.load(model_path)


def load_las_data(las_path: Path) -> pd.DataFrame:
    """Load LAS file and return DataFrame with required features."""
    las = lasio.read(las_path)
    df = las.df().reset_index()
    return df[FEATURE_COLUMNS]


def predict_fluid(df: pd.DataFrame, model) -> pd.DataFrame:
    """Make predictions and add labels to DataFrame."""
    df = df.copy()
    df['PREDICTION'] = model.predict(df)
    df['LABEL'] = df['PREDICTION'].apply(
        lambda x: LABELS[x] if 0 <= x < len(LABELS) else 'Unknown'
    )
    return df


def save_predictions(df: pd.DataFrame, output_path: Path) -> None:
    """Save predictions to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def main():
    # File paths
    las_file = BASE_DIR / 'P:/project\pythonpro\myvenv/fl-analysis/data/raw/OH Log/SJ-4RD1_Petrophysical Log.las'
    output_file = BASE_DIR / 'data/processed/SJ-4_Provosional.csv'
    
    # Load model and data
    model = load_model(MODEL_PATH)
    df = load_las_data(las_file)
    
    # Make predictions
    df_pred = predict_fluid(df, model)
    print(df_pred.head())
    
    # Save results
    save_predictions(df_pred, output_file)


if __name__ == "__main__":
    main()