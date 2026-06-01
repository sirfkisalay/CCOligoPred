"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 2: Hyperparameter Optimization of Selected Multiclass Models
==============================================================================
Description:
Performs Bayesian hyperparameter optimization (via Optuna) on the top 4 
performing ensemble classifiers (XGBoost, LightGBM, GradientBoosting, and 
ExtraTrees) utilizing the Register-Based Features (RBF) dataset.

Methodology:
- Search Algorithm: Tree-structured Parzen Estimator (TPE) via Optuna
- Cross-Validation: 5-Fold Stratified CV
- Target Metric: Overall Matthews Correlation Coefficient (MCC)
- Iterations: 60 Trials per model
==============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

# Model Imports
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
import optuna

# Suppress warnings for clean console output during Optuna execution
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING) # Keeps output clean, only shows final results

# =============================================================================
# --- Configuration & Dynamic Paths ---
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS")

# Utilizing the RBF Feature Set from Benchmarking
TRAIN_FILE_PATH = os.path.join(DATA_DIR, "MULTICLASS_RBF_TRAIN.xlsx")
TEST_FILE_PATH = os.path.join(DATA_DIR, "MULTICLASS_RBF_TEST.xlsx")

TARGET_COLUMN_NAME = 'Class'
RANDOM_STATE = 42
N_TRIALS = 60  
CV_FOLDS = 5

# =============================================================================
# --- Helper Functions ---
# =============================================================================
def multiclass_report(y_true, y_pred, model_name, encoder):
    """Calculates and prints publication-grade metrics including MOSS."""
    print(f"\n" + "="*60)
    print(f"--- Final Test Performance: {model_name} ---")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=encoder.classes_, digits=4, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    
    overall_mcc = matthews_corrcoef(y_true, y_pred)
    print(f"\nOverall MCC      : {overall_mcc:.4f}")
    
    mcc_per_class = {}
    print("Per-Class MCC:")
    for i, class_name in enumerate(encoder.classes_):
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        cls_mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
        mcc_per_class[class_name] = cls_mcc
        print(f"  - {str(class_name):<4} MCC : {cls_mcc:.4f}")
        
    tri_mcc = mcc_per_class.get("TRI", 0.0)
    tet_mcc = mcc_per_class.get("TET", 0.0)
    moss_score = (overall_mcc + tri_mcc + tet_mcc) / 3
    print(f"MOSS SCORE       : {moss_score:.4f}\n")
    return overall_mcc

def make_pipeline(model, impute=True):
    """Constructs the standard robust pipeline preventing data leakage."""
    steps = []
    if impute:
        steps.append(('imputer', KNNImputer(n_neighbors=5)))
    steps.append(('scaler', StandardScaler()))
    steps.append(('classifier', model))
    return Pipeline(steps)

# =============================================================================
# --- Load and Prepare Data ---
# =============================================================================
print("[*] Loading RBF Datasets...")
if not os.path.exists(TRAIN_FILE_PATH) or not os.path.exists(TEST_FILE_PATH):
    print(f"[!] ERROR: Datasets not found in {DATA_DIR}. Please ensure files exist.")
    exit()

train_df = pd.read_excel(TRAIN_FILE_PATH)
test_df = pd.read_excel(TEST_FILE_PATH)
print(f" -> Train Shape: {train_df.shape}, Test Shape: {test_df.shape}\n")

y_train_raw = train_df[TARGET_COLUMN_NAME]
X_train = train_df.drop(columns=[TARGET_COLUMN_NAME])
y_test_raw = test_df[TARGET_COLUMN_NAME]
X_test = test_df.drop(columns=[TARGET_COLUMN_NAME])
X_test = X_test[X_train.columns] # Ensure column alignment

# Standardize Labels
class_name_map = {1: 'PD', 2: 'APD', 3: 'TRI', 4: 'TET'}
y_train_mapped = y_train_raw.map(class_name_map).fillna(y_train_raw)
y_test_mapped = y_test_raw.map(class_name_map).fillna(y_test_raw)

encoder = LabelEncoder()
y_train = pd.Series(encoder.fit_transform(y_train_mapped))
y_test = pd.Series(encoder.transform(y_test_mapped))
NUM_CLASSES = len(encoder.classes_)

# =============================================================================
# --- Optuna Objective Functions (Preserving Exact Search Spaces) ---
# =============================================================================

# 1. XGBoost
def optuna_xgb_objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'objective': 'multi:softmax',
        'num_class': NUM_CLASSES,
        'eval_metric': 'mlogloss'
    }
    model = xgb.XGBClassifier(**param)
    pipeline = make_pipeline(model, impute=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    mccs = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = pipeline.predict(X_train.iloc[val_idx])
        mccs.append(matthews_corrcoef(y_train.iloc[val_idx], preds))
    return np.mean(mccs)

# 2. LightGBM
def optuna_lgb_objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'objective': 'multiclass',
        'num_class': NUM_CLASSES,
        'verbosity': -1
    }
    model = lgb.LGBMClassifier(**param)
    pipeline = make_pipeline(model, impute=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    mccs = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = pipeline.predict(X_train.iloc[val_idx])
        mccs.append(matthews_corrcoef(y_train.iloc[val_idx], preds))
    return np.mean(mccs)

# 3. GradientBoosting
def optuna_gb_objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': RANDOM_STATE
    }
    model = GradientBoostingClassifier(**param)
    pipeline = make_pipeline(model, impute=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    mccs = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = pipeline.predict(X_train.iloc[val_idx])
        mccs.append(matthews_corrcoef(y_train.iloc[val_idx], preds))
    return np.mean(mccs)

# 4. ExtraTrees
def optuna_et_objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
        'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30, 40, 50]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    }
    
    model = ExtraTreesClassifier(**param)
    pipeline = make_pipeline(model, impute=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    mccs = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = pipeline.predict(X_train.iloc[val_idx])
        mccs.append(matthews_corrcoef(y_train.iloc[val_idx], preds))
    return np.mean(mccs)

# =============================================================================
# --- Run Optimization & Final Evaluation ---
# =============================================================================
def run_study_and_evaluate(obj_fn, model_class, name, extra_params={}):
    print(f"\n[*] Running Optuna Optimization for {name} ({N_TRIALS} Trials)...")
    study = optuna.create_study(direction="maximize", study_name=name)
    study.optimize(obj_fn, n_trials=N_TRIALS, n_jobs=1)
    
    print(f" -> Best CV MCC: {study.best_trial.value:.4f}")
    print(f" -> Best Params: {study.best_trial.params}")
    
    # Merge best params with any extra required params (like random_state)
    final_params = {**study.best_trial.params, **extra_params}
    
    # Build Final Model
    print(f"[*] Training Final {name} Model on full Training Set...")
    model = model_class(**final_params)
    pipe = make_pipeline(model, impute=True)
    
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    # Print Report
    multiclass_report(y_test, preds, f"{name} (Optimized)", encoder)

# Execute the pipeline for all 4 models sequentially
if __name__ == "__main__":
    run_study_and_evaluate(optuna_xgb_objective, xgb.XGBClassifier, "XGBoost", 
                           {'random_state': RANDOM_STATE, 'n_jobs': -1, 'objective': 'multi:softmax', 'num_class': NUM_CLASSES, 'eval_metric': 'mlogloss'})
    
    run_study_and_evaluate(optuna_lgb_objective, lgb.LGBMClassifier, "LightGBM", 
                           {'random_state': RANDOM_STATE, 'n_jobs': -1, 'objective': 'multiclass', 'num_class': NUM_CLASSES, 'verbosity': -1})
    
    run_study_and_evaluate(optuna_gb_objective, GradientBoostingClassifier, "GradientBoosting", 
                           {'random_state': RANDOM_STATE})
    
    run_study_and_evaluate(optuna_et_objective, ExtraTreesClassifier, "ExtraTrees", 
                           {'random_state': RANDOM_STATE, 'n_jobs': -1})
    
    print("\n✅ Hyperparameter Tuning Protocol Complete.")