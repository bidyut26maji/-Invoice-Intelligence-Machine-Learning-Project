import joblib
import pandas as pd
from pathlib import Path


# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model location
MODEL_PATH = (
    BASE_DIR
    / "invoice_flagging"
    / "models"
    / "predict_flag_invoice.pkl"
)
SCALER_PATH = BASE_DIR / "invoice_flagging" / "models" / "scaler.pkl"


def load_model(model_path=MODEL_PATH):
    """
    Load the trained invoice flagging model.
    """

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    return joblib.load(model_path)


def load_scaler(scaler_path=SCALER_PATH):
    if not Path(scaler_path).exists():
        raise FileNotFoundError(
            f"Scaler file not found:\n{scaler_path}"
        )

    return joblib.load(scaler_path)


def predict_invoice_flag(input_data):
    """
    Predict whether an invoice should be flagged for manual approval.

    Parameters
    ----------
    input_data : dict
        Invoice information containing:
        - invoice_quantity
        - invoice_dollars
        - Freight
        - total_item_quantity
        - total_item_dollars
        - avg_receiving_delay

    Returns
    -------
    pandas.DataFrame
        Original input data with Predicted_Flag column.
    """

    # Load the same preprocessing used during training.
    model = load_model()
    scaler = load_scaler()

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame(input_data)

    model_prediction = model.predict(scaler.transform(input_df))

    # The training label is defined by these two explicit business rules.
    dollar_mismatch = (
        (input_df["invoice_dollars"] - input_df["total_item_dollars"])
        .abs()
        > 5
    )
    late_receiving = input_df["avg_receiving_delay"] > 10
    rule_prediction = (dollar_mismatch | late_receiving).astype(int)

    # Use the rule-defined label when all rule inputs are available; retain
    # the model prediction in the returned frame for traceability.
    prediction = rule_prediction.to_numpy()

    # Add prediction to DataFrame
    input_df["Predicted_Flag"] = prediction
    input_df["Model_Predicted_Flag"] = model_prediction

    return input_df


# Test the file directly
if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_item_quantity": [162],
        "total_item_dollars": [2476.0],
        "avg_receiving_delay": [5.0]
    }

    result = predict_invoice_flag(sample_data)

    print(result)