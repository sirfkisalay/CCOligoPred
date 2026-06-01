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
# Note: Changed 'TETRA' to 'TET' to match the updated scientific nomenclature from the manuscript
CLASS_NAMES = ['PD', 'APD', 'TRI', 'TET']

# NEW: OOF-Optimized Biophysical Thresholds for Margin Scaling
THRESHOLDS = np.array([0.35, 0.40, 0.45, 0.15])

def predict_multiclass(feature_df: pd.DataFrame) -> list:
    """
    Filters features using the Multi_GB list, applies margin-scaled 
    thresholds to XGBoost probabilities, and predicts multiclass states.
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
    
    # 3. Predict Raw Soft Probabilities
    raw_probs = model.predict_proba(X_filtered)
    
    # 4. Apply OOF Thresholds via Margin Scaling (Probability / Threshold)
    scaled_probs = raw_probs / THRESHOLDS
    preds = np.argmax(scaled_probs, axis=1)
    
    # 5. Map numeric predictions to string labels
    return [CLASS_NAMES[p] for p in preds]