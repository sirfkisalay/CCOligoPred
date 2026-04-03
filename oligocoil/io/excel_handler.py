import pandas as pd
import os

def load_sequence_data(file_path: str) -> pd.DataFrame:
    """
    Loads user data, auto-detects CSV vs Excel formats, and auto-fixes missing headers.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] Cannot find the input file at: {file_path}")
        
    # 1. Auto-detect format (Try Excel first, fallback to CSV)
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception:
        try:
            # If Excel fails, it's likely a CSV saved with an .xlsx extension
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to read the file as Excel or CSV. Error: {str(e)}")
            
    # 2. Auto-fix missing headers
    lower_cols = [str(c).lower() for c in df.columns]
    
    if 'sequence' not in lower_cols or 'register' not in lower_cols:
        # If it exactly has 2 columns but wrong names, assume headers are missing
        if len(df.columns) == 2:
            print("    -> [WARNING] 'Sequence' and 'Register' headers not found. Auto-assigning them to columns 1 and 2.")
            # Re-read without headers so we don't lose the first sequence!
            try:
                df = pd.read_excel(file_path, engine='openpyxl', header=None)
            except:
                df = pd.read_csv(file_path, header=None)
            df.columns = ['Sequence', 'Register']
        else:
            raise KeyError(f"[ERROR] Input file must contain 'Sequence' and 'Register' columns. Found: {list(df.columns)}")
    else:
        # Ensure exact capitalization for the rest of the pipeline
        col_map = {c: 'Sequence' for c in df.columns if str(c).lower() == 'sequence'}
        col_map.update({c: 'Register' for c in df.columns if str(c).lower() == 'register'})
        df = df.rename(columns=col_map)
        
    return df

def save_predictions(original_df: pd.DataFrame, multiclass_preds: list, binary_preds: list, binary_probs: list, output_path: str):
    """
    Appends the model predictions to the original dataframe and saves to a true Excel file.
    """
    result_df = original_df.copy()
    
    # Append the new prediction columns
    result_df['Multiclass_Prediction'] = multiclass_preds
    result_df['TRI_Binary_Prediction'] = binary_preds
    result_df['TRI_Probability'] = binary_probs
    
    # Always save the output as a true .xlsx file
    result_df.to_excel(output_path, index=False, engine='openpyxl')
