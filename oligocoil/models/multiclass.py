import os
import json
import joblib
import pandas as pd
import numpy as np

# Dynamically locate the weights folder inside the installed package
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
MODEL_PATH = os.path.join(WEIGHTS_DIR, 'oligocoil_multiclass_model.pkl')
FEATURES_PATH = os.path.join(WEIGHTS_DIR, 'MULTICLASS_ANOVA_Features.json')

CLASS_NAMES = ['PD', 'APD', 'TRI', 'TETRA']

def predict_multiclass(feature_df: pd.DataFrame) -> list:
    """Filters features using the ANOVA list and predicts multiclass states."""
    
    # 1. Load Model and Features
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model weight file: {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing feature list: {FEATURES_PATH}")
        
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        anova_features = json.load(f)
        
    # 2. Filter the incoming dataframe to only the ANOVA features
    # Ensure all required features are present
    missing = [f for f in anova_features if f not in feature_df.columns]
    if missing:
        raise ValueError(f"Feature matrix is missing {len(missing)} required ANOVA features.")
        
    X_filtered = feature_df[anova_features]
    
    # 3. Predict
    probs = model.predict_proba(X_filtered)
    preds = np.argmax(probs, axis=1)
    
    # Map numeric predictions to string labels (PD, APD, TRI, TETRA)
    return [CLASS_NAMES[p] for p in preds]
