"""
==============================================================================
CCOligoPred Manuscript Reproducibility
Step 9: Binary Hyperparameter Optimization (Optuna)
==============================================================================
Description:
Performs Bayesian hyperparameter optimization (via Optuna) on the top 4 
performing baseline classifiers (RandomForest, XGBoost, LightGBM, GradientBoosting) 
for the binary classification task (Target: State_tri).

Methodology:
- Feature Set: Register-Based Features (RBF).
- Validation: 5-Fold Stratified CV to prevent leakage.
- Preprocessing: KNNImputation (n_neighbors=5) -> Standard Scaling.
- Metric: Matthews Correlation Coefficient (MCC).
- Outputs: Optimizes models, evaluates on the test set, and exports 
  '09_Binary_Tuned_Models_Evaluation.csv'.
==============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd

# ML libraries
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Boosters
import xgboost as xgb
import lightgbm as lgb

# Bayesian optimization
import optuna

# Suppress warnings and Optuna spam for clean console output
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# --- 1. CONFIGURATION & DYNAMIC PATHS ---
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATASETS", "binary")

# Using the RBF feature set for optimization
TRAIN_FILE_PATH = os.path.join(DATA_DIR, "BINARY_RBF_TRAIN.xlsx")
TEST_FILE_PATH = os.path.join(DATA_DIR, "BINARY_RBF_TEST.xlsx")

TARGET_COLUMN_NAME = 'State_tri'
RANDOM_STATE = 42
N_TRIALS = 60  
CV_FOLDS = 5

# =============================================================================
# --- 2. HELPER FUNCTIONS ---
# =============================================================================
def make_pipeline(model, impute=True):
    """Leakage-proof pipeline containing imputation and scaling."""
    steps = []
    if impute:
        # Strict protocol requirement: KNNImputer
        steps.append(('imputer', KNNImputer(n_neighbors=5)))
    steps.append(('scaler', StandardScaler()))
    steps.append(('classifier', model))
    return Pipeline(steps)

def binary_evaluation_report(y_true, y_pred, model_name):
    """Prints and returns comprehensive binary metrics."""
    print(f"\n" + "="*50)
    print(f"--- Final Test Performance: {model_name} ---")
    print("="*50)
    
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate Sensitivity and Specificity
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sens, spec = np.nan, np.nan

    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print(f"Confusion Matrix:\n{cm}")
    print(f"\nAccuracy    : {acc:.4f}")
    print(f"MCC         : {mcc:.4f}")
    print(f"Sensitivity : {sens:.4f}")
    print(f"Specificity : {spec:.4f}\n")
    
    return {"Model": model_name, "Accuracy": acc, "MCC": mcc, "Sensitivity": sens, "Specificity": spec}

# =============================================================================
# --- 3. DATA PROCESSING ---
# =============================================================================
print(f"[*] Loading Binary RBF Datasets from: {DATA_DIR}")
if not os.path.exists(TRAIN_FILE_PATH) or not os.path.exists(TEST_FILE_PATH):
    print(f"[!] ERROR: Datasets not found in {DATA_DIR}. Please check the files.")
    exit()

train_df = pd.read_excel(TRAIN_FILE_PATH)
test_df = pd.read_excel(TEST_FILE_PATH)
print(f" -> Train Shape: {train_df.shape}, Test Shape: {test_df.shape}\n")

y_train = train_df[TARGET_COLUMN_NAME]
X_train = train_df.drop(columns=[TARGET_COLUMN_NAME])
y_test = test_df[TARGET_COLUMN_NAME]
X_test = test_df.drop(columns=[TARGET_COLUMN_NAME])

# Ensure column consistency
X_test = X_test[X_train.columns]

# =============================================================================
# --- 4. OPTUNA OBJECTIVE FUNCTIONS ---
# =============================================================================

# 1. RandomForest
def optuna_rf_objective(trial):
    param = {
        'n_estimators': trial.suggest_int("n_estimators", 200, 1500, step=100),
        'max_depth': trial.suggest_int("max_depth", 3, 30),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 10),
        'max_features': trial.suggest_categorical("max_features", ['sqrt', 'log2', 0.2, 0.5, None]),
        'n_jobs': -1, 
        'random_state': RANDOM_STATE, 
        'class_weight': 'balanced'
    }
    model = RandomForestClassifier(**param)
    pipeline = make_pipeline(model, impute=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    mccs = [matthews_corrcoef(y_train.iloc[val_idx], pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx]).predict(X_train.iloc[val_idx])) 
            for train_idx, val_idx in cv.split(X_train, y_train)]
    return np.mean(mccs)

# 2. XGBoost
def optuna_xgb_objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'random_state': RANDOM_STATE,
        'verbosity': 0,
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    model = xgb.XGBClassifier(**param)
    pipeline = make_pipeline(model, impute=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    mccs = [matthews_corrcoef(y_train.iloc[val_idx], pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx]).predict(X_train.iloc[val_idx])) 
            for train_idx, val_idx in cv.split(X_train, y_train)]
    return np.mean(mccs)

# 3. LightGBM
def optuna_lgb_objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'objective': 'binary',
        'verbosity': -1
    }
    model = lgb.LGBMClassifier(**param)
    pipeline = make_pipeline(model, impute=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    mccs = [matthews_corrcoef(y_train.iloc[val_idx], pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx]).predict(X_train.iloc[val_idx])) 
            for train_idx, val_idx in cv.split(X_train, y_train)]
    return np.mean(mccs)

# 4. GradientBoosting
def optuna_gb_objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': RANDOM_STATE
    }
    model = GradientBoostingClassifier(**param)
    pipeline = make_pipeline(model, impute=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    mccs = [matthews_corrcoef(y_train.iloc[val_idx], pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx]).predict(X_train.iloc[val_idx])) 
            for train_idx, val_idx in cv.split(X_train, y_train)]
    return np.mean(mccs)

# =============================================================================
# --- 5. EXECUTE OPTIMIZATION ---
# =============================================================================
def run_study(obj_fn, name):
    print(f"[*] Optimizing {name} ({N_TRIALS} Trials)...")
    study = optuna.create_study(direction="maximize", study_name=name)
    study.optimize(obj_fn, n_trials=N_TRIALS, n_jobs=1)
    print(f" -> Best CV MCC: {study.best_trial.value:.4f}")
    return study.best_trial.params

print("="*60)
print(" STARTING HYPERPARAMETER TUNING ")
print("="*60)

best_rf_params = run_study(optuna_rf_objective, "RandomForest")
best_xgb_params = run_study(optuna_xgb_objective, "XGBoost")
best_lgb_params = run_study(optuna_lgb_objective, "LightGBM")
best_gb_params = run_study(optuna_gb_objective, "GradientBoosting")

# =============================================================================
# --- 6. FINAL TESTING & EVALUATION ---
# =============================================================================
print("\n" + "="*80)
print(" FINAL EVALUATION OF OPTIMIZED MODELS ON UNSEEN TEST SET ")
print("="*80)

final_results = []

# Map of the optimized models
optimized_models = {
    "RandomForest": RandomForestClassifier(**best_rf_params, n_jobs=-1, random_state=RANDOM_STATE, class_weight='balanced'),
    "XGBoost": xgb.XGBClassifier(**best_xgb_params, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1, objective='binary:logistic', eval_metric='logloss'),
    "LightGBM": lgb.LGBMClassifier(**best_lgb_params, random_state=RANDOM_STATE, n_jobs=-1, objective='binary', verbosity=-1),
    "GradientBoosting": GradientBoostingClassifier(**best_gb_params, random_state=RANDOM_STATE)
}

# Train, Predict, Evaluate Loop
for name, model in optimized_models.items():
    pipe = make_pipeline(model, impute=True)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    metrics = binary_evaluation_report(y_test, preds, f"{name} (Optimized)")
    final_results.append(metrics)

# Save Final Results
results_df = pd.DataFrame(final_results)
csv_filename = os.path.join(BASE_DIR, "09_Binary_Tuned_Models_Evaluation.csv")
results_df.to_csv(csv_filename, index=False)

print(f"✅ Final evaluated metrics have been saved to:\n{csv_filename}")