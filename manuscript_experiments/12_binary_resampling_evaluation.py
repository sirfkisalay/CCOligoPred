"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 12: Binary Evaluation Protocol (Resampling Comparison)
==============================================================================
Description:
Executes a comprehensive combinatorial evaluation of 36 pipelines (3 Feature 
Sets x 3 Resamplers x 4 Models) to address class imbalance for the binary 
prediction task (Target: Trimer vs. Rest).

Resampling Techniques:
1. SMOTE (Synthetic Minority Over-sampling Technique)
2. SMOTE-Tomek (Hybrid: Oversampling + Tomek Links undersampling)
3. SMOTE-ENN (Hybrid: Oversampling + Edited Nearest Neighbors)

Methodology:
- Pipeline: Impute -> Scale -> Resample -> Evaluate (Imblearn Pipeline)
- Metric: Global MCC, Per-Class MCC (Rest, Tri)
- Output: '12_Binary_Resampling_Results.csv'
==============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter
import warnings

# Sklearn & Metrics
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

# Resampling & Pipeline (CRITICAL for No Leakage)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline 

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
# Dynamically locate the DATASETS/binary folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS", "binary")

TRAIN_FILE = "BINARY_RBF_TRAIN.xlsx"
TEST_FILE  = "BINARY_RBF_TEST.xlsx"
TARGET_COLUMN = "State_tri"
CLASS_NAMES = ['Rest', 'Tri'] # 0, 1
RANDOM_STATE = 42

# Feature Set Filenames (Generated from Step 10)
FEATURE_FILES = {
    "Binary_ANOVA": "Binary_ANOVA_Features.json",
    "Binary_LGBM":  "Binary_LGBM_Importance_Features.json",
    "Binary_RFECV": "Binary_RFECV_Features.json"
}

# --- OPTIMIZED HYPERPARAMETERS (Binary) ---
rf_params = {
    'n_estimators': 1300, 'max_depth': 23, 'min_samples_leaf': 9, 'max_features': 0.5,
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': 0
}

xgb_params = {
    'n_estimators': 1150, 'max_depth': 12, 'learning_rate': 0.0405246, 
    'subsample': 0.75433, 'colsample_bytree': 0.60561,
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0,
    'objective': 'binary:logistic', 'eval_metric': 'logloss'
}

lgb_params = {
    'n_estimators': 860, 'learning_rate': 0.046423, 'num_leaves': 30, 'max_depth': 13,
    'subsample': 0.80635, 'colsample_bytree': 0.67299, 
    'reg_alpha': 1.1078e-08, 'reg_lambda': 0.17206,
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1,
    'objective': 'binary'
}

gb_params = {
    'n_estimators': 1350, 'learning_rate': 0.021942, 'max_depth': 9, 
    'min_samples_leaf': 8, 'subsample': 0.96444, 'max_features': None,
    'random_state': RANDOM_STATE
}

models_to_train = {
    "RandomForest": RandomForestClassifier(**rf_params),
    "XGBoost": xgb.XGBClassifier(**xgb_params),
    "LightGBM": lgb.LGBMClassifier(**lgb_params),
    "GradientBoosting": GradientBoostingClassifier(**gb_params)
}

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def load_feature_list(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_per_class_mcc(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    res = {}
    for i, cls_name in enumerate(classes):
        # Binary specific: 0=TN, 1=TP logic applies, but One-vs-Rest calculation remains valid
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - (tp + fp + fn))
        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = (tp * tn - fp * fn) / denom if denom != 0 else 0.0
        res[cls_name] = mcc
    return res

# ==============================================================================
# 3. DATA LOADING & RESAMPLER SETUP
# ==============================================================================
print(f"[*] Loading datasets from {DATA_DIR}...")
train_df = pd.read_excel(os.path.join(DATA_DIR, TRAIN_FILE))
test_df  = pd.read_excel(os.path.join(DATA_DIR, TEST_FILE))

y_train_full = train_df[TARGET_COLUMN].astype(int)
y_test_full  = test_df[TARGET_COLUMN].astype(int)

# Calculate safe k_neighbors based on minority class
min_class_size = Counter(y_train_full).most_common()[-1][1]
k_neigh = min(5, max(1, min_class_size - 1))
print(f" -> Minority Class Size: {min_class_size}. Using safe k_neighbors={k_neigh}\n")

# Define Resampling Strategies
base_smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neigh)

resampling_strategies = {
    "SMOTE": base_smote,
    "SMOTE_Tomek": SMOTETomek(random_state=RANDOM_STATE, smote=base_smote),
    "SMOTE_ENN": SMOTEENN(random_state=RANDOM_STATE, smote=base_smote)
}

# ==============================================================================
# 4. EXECUTION LOOP
# ==============================================================================
results_log = []

print("="*100)
print(f"{'FEAT':<15} | {'RESAMPLER':<12} | {'MODEL':<16} | {'G-MCC':<6} | {'PER CLASS MCC (Rest, Tri)'}")
print("="*100)

# Loop 1: Feature Sets
for feat_name, json_file in FEATURE_FILES.items():
    
    # Load features safely
    json_path = os.path.join(DATA_DIR, json_file)
    if not os.path.exists(json_path): 
        print(f"[!] Warning: Missing {json_file}. Skipping {feat_name}...")
        continue
        
    try:
        selected_feats = load_feature_list(json_path)
        valid_feats = [f for f in selected_feats if f in train_df.columns]
        
        X_train_sub = train_df[valid_feats]
        X_test_sub  = test_df[valid_feats]
    except Exception as e:
        print(f"[!] Error processing {feat_name}: {e}")
        continue

    # Loop 2: Resampling Methods
    for res_name, res_method in resampling_strategies.items():
        
        # Loop 3: Models
        for model_name, model_inst in models_to_train.items():
            
            # Construct Pipeline (Leakage Proof)
            pipeline = ImbPipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('resampler', res_method), 
                ('clf', clone(model_inst))
            ])
            
            # Train
            pipeline.fit(X_train_sub, y_train_full)
            
            # Predict (SMOTE is ignored here during testing by ImbPipeline)
            y_pred = pipeline.predict(X_test_sub)
            
            # Evaluate
            global_mcc = matthews_corrcoef(y_test_full, y_pred)
            pc_mcc = calculate_per_class_mcc(y_test_full, y_pred, CLASS_NAMES)
            
            # Log
            res_entry = {
                'Feature_Method': feat_name,
                'Resampler': res_name,
                'Model': model_name,
                'Global_MCC': global_mcc,
                **{f"MCC_{k}": v for k, v in pc_mcc.items()}
            }
            results_log.append(res_entry)
            
            # Print Row
            pc_str = ", ".join([f"{v:.3f}" for v in pc_mcc.values()])
            print(f"{feat_name:<15} | {res_name:<12} | {model_name:<16} | {global_mcc:.4f} | [{pc_str}]")

print("="*100)

# ==============================================================================
# 5. SAVE & REPORT
# ==============================================================================
results_df = pd.DataFrame(results_log)
out_file = os.path.join(BASE_DIR, "12_Binary_Resampling_Results.csv")
results_df.to_csv(out_file, index=False)

print(f"\n✅ Detailed results successfully saved to:\n{out_file}")

best_run = results_df.loc[results_df['Global_MCC'].idxmax()]
print("\n🏆 BEST PERFORMING CONFIGURATION (Binary Task) 🏆")
print(f"Feature Set : {best_run['Feature_Method']}")
print(f"Resampler   : {best_run['Resampler']}")
print(f"Model       : {best_run['Model']}")
print(f"Global MCC  : {best_run['Global_MCC']:.4f}")
print(f"Tri MCC     : {best_run['MCC_Tri']:.4f}")
print(f"Rest MCC    : {best_run['MCC_Rest']:.4f}")