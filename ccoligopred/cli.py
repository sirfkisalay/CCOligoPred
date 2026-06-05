import argparse
import sys
import warnings
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Import your custom modules
from ccoligopred.io.excel_handler import load_sequence_data, save_predictions
from ccoligopred.rbf import generate_all_features
from ccoligopred.models import predict_multiclass, predict_binary

# =========================================================================
# MARGIN SCALING THRESHOLD OPTIMIZATION
# =========================================================================
def apply_threshold_optimization(multiclass_probs):
    # Model now outputs native columns [0, 1, 2, 3] corresponding to [1, 2, 3, 4]
    # True Journal mapping: 1=APD, 2=PD, 3=TRI, 4=TET
    class_mapping = {0: 'APD', 1: 'PD', 2: 'TRI', 3: 'TET'}
    
    # Applied exactly to the raw columns: [APD, PD, TRI, TET]
    THRESHOLDS = np.array([0.35, 0.40, 0.45, 0.15])
    
    scaled_probs = multiclass_probs / THRESHOLDS
    predicted_indices = np.argmax(scaled_probs, axis=1)
    
    final_classes = [class_mapping[idx] for idx in predicted_indices]
    final_confidences = [multiclass_probs[i, idx] for i, idx in enumerate(predicted_indices)]
            
    return final_classes, final_confidences
def main():
    parser = argparse.ArgumentParser(description="CCOligoPred")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Predict Command
    parser_predict = subparsers.add_parser("predict")
    parser_predict.add_argument("-i", "--input", required=True)
    parser_predict.add_argument("-o", "--output", default="ccoligopred_predictions.xlsx")

    # RBF Command
    parser_rbf = subparsers.add_parser("rbf")
    parser_rbf.add_argument("-i", "--input", required=True)
    parser_rbf.add_argument("-o", "--output", default="ccoligopred_rbf_features.xlsx")

    args = parser.parse_args()
    
    try:
        df = load_sequence_data(args.input)
        feature_matrix = generate_all_features(df)

        if args.command == "rbf":
            if isinstance(feature_matrix, pd.DataFrame):
                final_rbf_df = pd.concat([df, feature_matrix], axis=1)
            else:
                final_rbf_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(feature_matrix).reset_index(drop=True)], axis=1)
            final_rbf_df.to_excel(args.output, index=False)
            print(f"RBF extraction saved to: {args.output}")
            return

        elif args.command == "predict":
            binary_preds, binary_probs = predict_binary(feature_matrix)
            raw_multiclass_preds, multiclass_probs = predict_multiclass(feature_matrix)
            
            # Apply Margin Scaling Logic
            final_multi_classes, final_multi_probs = apply_threshold_optimization(multiclass_probs)
            
            save_predictions(df, final_multi_classes, final_multi_probs, multiclass_probs, binary_preds, binary_probs, args.output)
            print(f"Prediction complete. Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()