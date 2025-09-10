import os
import io
import re
import joblib
import shap
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import redirect_stdout, redirect_stderr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# ------------------------------
# Download and load artifacts from Hugging Face Hub
# ------------------------------
HF_REPO = "Dinesh2001/Llama3.2-1B-QLoRA-Explainer"

# Scaler
scaler_path = hf_hub_download(repo_id=HF_REPO, filename="numerical_scaler.pkl")
scaler = joblib.load(scaler_path)

# Random Forest model
rf_model_path = hf_hub_download(repo_id=HF_REPO, filename="rf_model_pruned_compressed.joblib")
rf_model = joblib.load(rf_model_path)

distance_df = pd.read_csv("order_city_country_distance.csv")
performance_scores = pd.read_excel("performance_scores.xlsx", sheet_name=None)

# LLaMA GGUF model
llm_path = hf_hub_download(repo_id=HF_REPO, filename="llama32-1b-merged-Q4_K_M.gguf")
llm = Llama(model_path=llm_path, n_threads=8, n_ctx=2048, verbose=False)

# ------------------------------
# Configs
# ------------------------------
CLASS_NAMES = {0: "early", 1: "on_time", 2: "delay"}

shipping_map = {
    'Standard Class': 1,
    'Second Class': 2,
    'First Class': 3,
    'Same Day': 4
}
shipping_mode_to_days = {
    'Same Day': 0,
    'First Class': 1,
    'Second Class': 2,
    'Standard Class': 4
}
payment_categories = ["CASH", "DEBIT", "PAYMENT", "TRANSFER"]

def get_daypart(hour):
    if 4 <= hour <= 7:
        return 'Early Morning'
    elif 8 <= hour <= 11:
        return 'Morning'
    elif 12 <= hour <= 15:
        return 'Noon'
    elif 16 <= hour <= 19:
        return 'Eve'
    elif 20 <= hour <= 23:
        return 'Night'
    else:
        return 'Late Night'

daypart_map = {
    'Early Morning': 0,
    'Morning': 1,
    'Noon': 2,
    'Eve': 3,
    'Night': 4,
    'Late Night': 5
}

# ------------------------------
# Request schema
# ------------------------------
class OrderInput(BaseModel):
    profit_per_order: float
    order_item_discount: float
    order_item_product_price: float
    order_item_profit_ratio: float
    order_item_quantity: int
    sales: float
    order_profit_per_order: float
    shipping_mode: str
    order: dict
    customer: dict
    payment_type: str
    order_datetime: str
    ship_datetime: str

# ------------------------------
# Preprocessing
# ------------------------------
def preprocess_input(input_data: dict) -> pd.DataFrame:
    row = {
        "profit_per_order": input_data["profit_per_order"],
        "order_item_discount": input_data["order_item_discount"],
        "order_item_product_price": input_data["order_item_product_price"],
        "order_item_profit_ratio": input_data["order_item_profit_ratio"],
        "order_item_quantity": input_data["order_item_quantity"],
        "sales": input_data["sales"],
        "order_profit_per_order": input_data["order_profit_per_order"],
    }

    shipping_mode = input_data["shipping_mode"]
    row["shipping_mode"] = shipping_map.get(shipping_mode, 0)

    # Distance
    city = input_data["order"]["city"]
    country = input_data["order"]["country"]
    dist_val = distance_df.loc[
        (distance_df["order_city"] == city) &
        (distance_df["order_country"] == country),
        "store_to_order_distance_km"
    ]
    if dist_val.empty:
        dist_val = distance_df["store_to_order_distance_km"].mean()
    else:
        dist_val = dist_val.values[0]
    row["distance_normalized"] = dist_val

    # Date features
    order_dt = datetime.fromisoformat(input_data["order_datetime"])
    ship_dt = datetime.fromisoformat(input_data["ship_datetime"])

    row["order_to_shipment_days"] = (ship_dt - order_dt).days
    row["order_shipping_time"] = (ship_dt - order_dt).total_seconds() / 3600.0
    row["order_to_shipment_planned_days"] = shipping_mode_to_days.get(shipping_mode, np.nan)
    row["shipment_delay_days"] = row["order_to_shipment_days"] - row["order_to_shipment_planned_days"]

    # Derived features
    order_full_location = (
        input_data["order"]["region"] + "|" +
        input_data["order"]["country"] + "|" +
        input_data["order"]["state"] + "|" +
        input_data["order"]["city"]
    )
    customer_full_location = (
        input_data["customer"]["country"] + "|" +
        input_data["customer"]["state"] + "|" +
        input_data["customer"]["city"]
    )
    derived_features = {
        "order_full_location": order_full_location,
        "customer_full_location": customer_full_location,
        "order_dayofweek": order_dt.weekday(),
        "shipping_dayofweek": ship_dt.weekday(),
        "order_hour": order_dt.hour,
        "shipping_hour": ship_dt.hour,
        "order_daynight": daypart_map[get_daypart(order_dt.hour)],
        "ship_daynight": daypart_map[get_daypart(ship_dt.hour)],
    }

    for feature, value in derived_features.items():
        sheet = performance_scores[feature]
        score_col = f"performance_score_{feature}"
        if value in sheet[feature].values:
            score = sheet.loc[sheet[feature] == value, score_col].values[0]
        else:
            score = sheet[score_col].mean()
        row[score_col] = score

    # Payment type one-hot
    payment_type = input_data["payment_type"]
    for cat in payment_categories:
        row[f"payment_type_{cat}"] = 1 if payment_type == cat else 0

    df_row = pd.DataFrame([row])
    numerical_features_to_scale = [
        "profit_per_order", "order_item_discount", "order_item_product_price",
        "order_item_profit_ratio", "order_item_quantity", "sales",
        "order_profit_per_order", "distance_normalized", "order_shipping_time",
        "order_to_shipment_days", "order_to_shipment_planned_days",
        "shipment_delay_days", "performance_score_order_full_location",
        "performance_score_customer_full_location",
        "performance_score_order_dayofweek", "performance_score_shipping_dayofweek",
        "performance_score_order_hour", "performance_score_shipping_hour",
        "performance_score_order_daynight", "performance_score_ship_daynight"
    ]
    df_row[numerical_features_to_scale] = scaler.transform(df_row[numerical_features_to_scale])
    return df_row

# ------------------------------
# SHAP explanation
# ------------------------------
def explain_prediction(model, X, class_names=CLASS_NAMES, num_features=2):
    proba = model.predict_proba(X)[0]
    pred_class_idx = np.argmax(proba)
    pred_label = class_names.get(pred_class_idx, str(pred_class_idx))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_for_pred = shap_values[pred_class_idx][0]
    elif shap_values.ndim == 3:
        shap_for_pred = shap_values[0, :, pred_class_idx]
    else:
        shap_for_pred = shap_values[0]

    feature_importance = dict(zip(X.columns, shap_for_pred.tolist()))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_positive = [{f: v} for f, v in sorted_features[:num_features]]
    top_negative = [{f: v} for f, v in sorted_features[-num_features:]]

    return {
        "predicted_class": pred_label,
        "predicted_probabilities": {class_names.get(i, str(i)): float(p) for i, p in enumerate(proba)},
        "top_positive_features": top_positive,
        "top_negative_features": top_negative
    }

# ------------------------------
# LLM explanation
# ------------------------------
def llm_explain(explanation: dict):
    prompt = f"Explain the below predictions:\n\n{explanation}"
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        output = llm(prompt, max_tokens=250, temperature=1.0, top_p=0.9)
    raw_text = output["choices"][0]["text"]
        # --- Clean the text ---
    cleaned = raw_text.strip()                          # remove leading/trailing whitespace
    cleaned = re.sub(r"^[^A-Za-z0-9]+", "", cleaned)    # remove junk at the start (like }; or \n)
    cleaned = re.sub(r"\s+", " ", cleaned)              # collapse multiple spaces/newlines

    return cleaned

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Delivery Delay Prediction API")

@app.post("/predict")
def predict(input_data: OrderInput):
    try:
        processed = preprocess_input(input_data.dict())
        explanation = explain_prediction(rf_model, processed)
        llm_text = llm_explain(explanation)

        return {
            "prediction": explanation["predicted_class"],
            # "explanation": explanation,
            # "probabilities": explanation["predicted_probabilities"],
            # "shap_features": {
            #     "top_positive": explanation["top_positive_features"],
            #     "top_negative": explanation["top_negative_features"]
            # },
            "llm_explanation": llm_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# {
#   "profit_per_order": 30,
#   "order_item_discount": 15,
#   "order_item_product_price": 100,
#   "order_item_profit_ratio": 0.1,
#   "order_item_quantity": 1,
#   "sales": 200,
#   "order_profit_per_order": 30,
#   "customer": {
#     "country": "EE. UU.",
#     "state": "CT",
#     "city": "Milford"
#   },
#   "order": {
#     "region": "Western Europe",
#     "country": "Austria",
#     "state": "Vienna",
#     "city": "Viena"
#   },
#   "payment_type": "CASH",
#   "shipping_mode": "Standard Class",
#   "order_datetime": "2025-09-08 21:17:38.575358",
#   "ship_datetime": "2025-09-10 23:17:38.575358"
# }