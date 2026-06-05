import os
import json
import pickle
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Dynamically locate the weights folder inside the installed package
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')

# UPDATED: Pointing to the new XGBoost model and the Gradient Boosting feature JSON
MODEL_PATH = os.path.join(WEIGHTS_DIR, 'ccoligopred_multiclass_model.pkl')
FEATURES_PATH = os.path.join(WEIGHTS_DIR, 'Multiclass_GB_Importance_Features_mean_Raw.json')

# CLASS_NAMES mapped to LabelEncoder (0: PD, 1: APD, 2: TRI, 3: TET)
CLASS_NAMES = ['PD', 'APD', 'TRI', 'TET']

def predict_multiclass(feature_df: pd.DataFrame):
    """
    Filters features using the Multi_GB list, predicts raw standard classes,
    and calculates the full probability matrix to be sent to the CLI for threshold optimization.
    """
    
    # 1. Load Model and Features Safely
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model weight file: {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing feature list: {FEATURES_PATH}")
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    with open(FEATURES_PATH, 'r') as f:
        gb_features = json.load(f)
        
    # 2. Filter the incoming dataframe to only the Multi_GB features
    missing = [f for f in gb_features if f not in feature_df.columns]
    if missing:
        raise ValueError(f"Feature matrix is missing {len(missing)} required Multi_GB features.")
        
    X_filtered = feature_df[gb_features]
    
    # 3. Predict Raw Hard Predictions (Standard XGBoost Argmax)
    raw_preds_numeric = model.predict(X_filtered)
    raw_preds_string = [CLASS_NAMES[p] for p in raw_preds_numeric]
    
    # 4. Predict Raw Soft Probabilities (Required for cli.py custom thresholds)
    raw_probs = model.predict_proba(X_filtered)
    
    # 5. Return BOTH so cli.py can unpack them properly
    return raw_preds_string, raw_probs