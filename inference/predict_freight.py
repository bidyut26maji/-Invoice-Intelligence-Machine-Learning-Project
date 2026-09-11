import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "freight_cost_prediction"
    / "models"
    / "predict_freight_model.pkl"
)


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict
        Input invoice data.

    Returns
    -------
    pd.DataFrame
        DataFrame with predicted freight cost.
    """

    model = load_model()

    input_df = pd.DataFrame(input_data)
    input_df = input_df[model.feature_names_in_]

    input_df["Predicted_Freight"] = model.predict(input_df).round()

    return input_df


if __name__ == "__main__":

    # Example inference run (local testing)
    sample_data = {
        "Dollars": [18500, 9000, 3000, 200]
    }

    prediction = predict_freight_cost(sample_data)

    print(prediction)