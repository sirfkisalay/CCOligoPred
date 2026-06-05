import pandas as pd

def load_sequence_data(input_path):
    """
    Loads sequence and register data from an Excel file.
    Ensures that the necessary columns exist.
    """
    try:
        df = pd.read_excel(input_path)
        
        # Strip hidden spaces from headers just to be safe
        df.columns = df.columns.str.strip()
        
        # Check for required columns
        if 'Sequence' not in df.columns or 'Register' not in df.columns:
            raise ValueError("Input Excel file must contain exactly 'Sequence' and 'Register' columns.")
            
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load input file {input_path}: {str(e)}")


def save_predictions(df, multi_classes, multi_confidences, raw_multi_probs, binary_preds, binary_probs, output_path):
    """
    Appends multiclass predictions, optimized confidences, raw probabilities, 
    and binary predictions to the original dataframe and saves it to Excel.
    """
    # Work on a copy to prevent fragmentation warnings
    out_df = df.copy()
    
    # 1. Final Multiclass Predictions & Confidence (After Threshold Optimization)
    out_df['Multiclass_Prediction'] = multi_classes
    out_df['Multiclass_Confidence_Score'] = multi_confidences
    
    # 2. Raw Multiclass Probabilities (Directly from XGBoost)
    out_df['PD_Prob'] = raw_multi_probs[:, 0]
    out_df['APD_Prob'] = raw_multi_probs[:, 1]
    out_df['TRI_Prob'] = raw_multi_probs[:, 2]
    out_df['TET_Prob'] = raw_multi_probs[:, 3]
    
    # 3. Binary Predictions & Confidence
    out_df['TRI_Binary_Prediction'] = binary_preds
    out_df['TRI_Binary_Confidence'] = binary_probs
    
    # Save the compiled dataframe
    out_df.to_excel(output_path, index=False)
    print(f"Successfully saved {len(out_df)} predictions to {output_path}")