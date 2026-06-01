"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 5: Multiclass Resampling Techniques Evaluation
==============================================================================
Description:
Executes a massive combinatorial evaluation of 72 pipelines to address extreme 
class imbalances (specifically for TRI and TET states). Tests 6 reduced feature 
subsets against 3 synthetic minority oversampling techniques across 4 optimized 
baseline classifiers.

Resampling Techniques:
1. SMOTE (Synthetic Minority Over-sampling Technique)
2. SMOTE-Tomek (Hybrid: Oversampling + Tomek Links undersampling)
3. SMOTE-ENN (Hybrid: Oversampling + Edited Nearest Neighbors)

Methodology:
- Pipeline: Impute -> Scale -> Resample -> Evaluate (Imblearn Pipeline)
- Metric: MOSS Score (prioritizing TRI/TET identification)
- Output: '05_Multiclass_Resampling_Results.csv'
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# Models
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

# Resampling & Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline 

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
# Dynamically locate the DATASETS folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS")

TRAIN_FILE = "MULTICLASS_RBF_TRAIN.xlsx"
TEST_FILE  = "MULTICLASS_RBF_TEST.xlsx"

TARGET_COLUMN = "Class"
CLASS_NAMES = ['PD', 'APD', 'TRI', 'TET'] 
RANDOM_STATE = 42

# Multiclass Feature Set Filenames (From Step 3)
FEATURE_FILES = {
    "Multi_RFECV": "Multiclass_RFECV_Features_Raw.json",
    "Multi_ANOVA": "Multiclass_ANOVA_Features_Raw.json",
    "Multi_ExtraTrees": "Multiclass_ExtraTrees_Importance_Features_Raw.json",
    "Multi_GB": "Multiclass_GB_Importance_Features_mean_Raw.json",
    "Multi_XGB": "Multiclass_XGB_Importance_Features_mean_Raw.json",
    "Multi_LGBM": "Multiclass_LGBM_Importance_Features_mean_Raw.json"
}

# --- OPTIMIZED HYPERPARAMETERS ---
xgb_params = {
    'n_estimators': 700, 'max_depth': 11, 'learning_rate': 0.0147951583299575, 
    'subsample': 0.7776832317133678, 'colsample_bytree': 0.5870074173438061, 
    'gamma': 0.9200227958551773, 'min_child_weight': 4,
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0,
    'objective': 'multi:softmax', 'eval_metric': 'mlogloss' 
}

lgb_params = {
    'n_estimators': 1250, 'max_depth': 23, 'learning_rate': 0.04472528989077509, 
    'num_leaves': 113, 'subsample': 0.9786298784381904, 'colsample_bytree': 0.8115594723131374, 
    'min_child_samples': 20, 'class_weight': 'balanced',
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1,
    'objective': 'multiclass', 'num_class': 4
}

gb_params = {
    'n_estimators': 1900, 'learning_rate': 0.18698360549487777, 'max_depth': 9, 
    'min_samples_split': 20, 'min_samples_leaf': 15, 'subsample': 0.8386502590161441, 
    'max_features': 'log2',
    'random_state': RANDOM_STATE
}

et_params = {
    'n_estimators': 600, 'max_depth': 50, 'min_samples_split': 15, 
    'min_samples_leaf': 2, 'max_features': None, 'bootstrap': True, 
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': 0
}

models_to_train = {
    "XGBoost": xgb.XGBClassifier(**xgb_params),
    "LightGBM": lgb.LGBMClassifier(**lgb_params),
    "GradientBoosting": GradientBoostingClassifier(**gb_params),
    "ExtraTrees": ExtraTreesClassifier(**et_params)
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
print(f"[*] Loading datasets from: {DATA_DIR}")
train_df = pd.read_excel(os.path.join(DATA_DIR, TRAIN_FILE))
test_df  = pd.read_excel(os.path.join(DATA_DIR, TEST_FILE))

y_train_raw = train_df[TARGET_COLUMN]
y_test_raw  = test_df[TARGET_COLUMN]

# Map targets consistently
class_name_map = {1: 'PD', 2: 'APD', 3: 'TRI', 4: 'TET'}
y_train_mapped = y_train_raw.map(class_name_map).fillna(y_train_raw)
y_test_mapped = y_test_raw.map(class_name_map).fillna(y_test_raw)

# Encode target labels (Maps PD, APD, TRI, TET to 0, 1, 2, 3)
le = LabelEncoder()
y_train_full = le.fit_transform(y_train_mapped)
y_test_full = le.transform(y_test_mapped)

# Calculate safe k_neighbors based on the absolute minority class across all 4 classes
min_class_size = Counter(y_train_full).most_common()[-1][1]
k_neigh = min(5, max(1, min_class_size - 1))
print(f" -> Minority Class Size: {min_class_size}. Using robust k_neighbors={k_neigh}")

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

print("\n" + "="*120)
print(f"{'FEAT':<18} | {'RESAMPLER':<12} | {'MODEL':<16} | {'G-MCC':<6} | {'MOSS':<6} | {'PER CLASS MCC ' + str(CLASS_NAMES)}")
print("="*120)

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
            
            # Construct Pipeline (Leakage Proof via Imblearn)
            pipeline = ImbPipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('resampler', res_method), 
                ('clf', clone(model_inst))
            ])
            
            # Train & Predict
            pipeline.fit(X_train_sub, y_train_full)
            y_pred = pipeline.predict(X_test_sub)
            
            # Evaluate Metrics
            global_mcc = matthews_corrcoef(y_test_full, y_pred)
            pc_mcc = calculate_per_class_mcc(y_test_full, y_pred, CLASS_NAMES)
            
            moss_score = (global_mcc + pc_mcc.get("TRI", 0.0) + pc_mcc.get("TET", 0.0)) / 3
            
            # Log Data
            res_entry = {
                'Feature_Method': feat_name,
                'Resampler': res_name,
                'Model': model_name,
                'Global_MCC': global_mcc,
                'MOSS_Score': moss_score,
                **{f"MCC_{k}": v for k, v in pc_mcc.items()}
            }
            results_log.append(res_entry)
            
            # Print Row
            pc_str = ", ".join([f"{v:.3f}" for v in pc_mcc.values()])
            print(f"{feat_name:<18} | {res_name:<12} | {model_name:<16} | {global_mcc:.4f} | {moss_score:.4f} | [{pc_str}]")

print("="*120)

# ==============================================================================
# 5. SAVE & REPORT
# ==============================================================================
results_df = pd.DataFrame(results_log)
out_file = os.path.join(BASE_DIR, "05_Multiclass_Resampling_Results.csv")
results_df.to_csv(out_file, index=False)

print(f"\nDetailed results saved to: {out_file}")

best_run = results_df.loc[results_df['MOSS_Score'].idxmax()]
print("\n🏆 BEST PERFORMING CONFIGURATION (By MOSS Score) 🏆")
print(f"Feature Set : {best_run['Feature_Method']}")
if 'Resampler' in best_run:
    print(f"Resampler   : {best_run['Resampler']}")
print(f"Model       : {best_run['Model']}")
print("-" * 35)
print(f"Global MCC  : {best_run['Global_MCC']:.4f}")
print(f"MOSS Score  : {best_run['MOSS_Score']:.4f}")
print("-" * 35)
print(f"PD MCC      : {best_run['MCC_PD']:.4f}")
print(f"APD MCC     : {best_run['MCC_APD']:.4f}")
print(f"TRI MCC     : {best_run['MCC_TRI']:.4f}")
print(f"TET MCC     : {best_run['MCC_TET']:.4f}")