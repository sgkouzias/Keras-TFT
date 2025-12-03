import pandas as pd
import numpy as np
import keras
import gc
from typing import List, Optional, Union
from .model import TFTForecaster
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def timeseries_cv_with_covariates(
    model: TFTForecaster, 
    df: pd.DataFrame, 
    num_windows: int,
    forecast_horizon: int = 7,
    target_col: str = 'y',
    past_cov_cols: Optional[List[str]] = None,
    future_cov_cols: Optional[List[str]] = None,
    epochs: int = 10,
    batch_size: int = 32,
    verbose: int = 0,
    scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None
):
    """
    Time series cross-validation with TFT and covariates.
    
    Automatically detects covariates from the dataframe if not explicitly provided.
    Handles different time frequencies by inferring from the dataframe index.
    Prints detailed performance metrics (RMSE, MAE, MAPE).
    
    The number of windows is fixed, and the start date is inferred to cover exactly
    num_windows * forecast_horizon steps at the end of the dataset.
    Stride is set equal to forecast_horizon (non-overlapping test sets).

    Args:
        model (TFTForecaster): Initialized TFTForecaster model.
        df (pd.DataFrame): DataFrame with DatetimeIndex or 'timestamp' column. 
                           Must contain the target column.
        num_windows (int): Number of cross-validation windows to perform.
        forecast_horizon (int, optional): Number of steps to forecast ahead. Defaults to 7.
        target_col (str, optional): Name of target column. Defaults to 'y'.
        past_cov_cols (List[str], optional): List of past covariate columns.
        future_cov_cols (List[str], optional): List of future covariate columns. 
                                               If None, defaults to all non-target columns.
        epochs (int, optional): Number of training epochs per window. Defaults to 10.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        verbose (int, optional): Verbosity mode. Defaults to 0.
        scaler (Union[MinMaxScaler, StandardScaler], optional): Scaler used to scale the target column. If provided, metrics will be calculated on inverse transformed data.

    Returns:
        pd.DataFrame: Results DataFrame containing 'id', 'timestamp', 'target_name', 'predictions'.
    """
    results = []
    metrics = {'rmse': [], 'mae': [], 'mape': []}
    
    # Validate target column
    if target_col not in df.columns:
        raise ValueError(f"DataFrame must contain a target column named '{target_col}'.")
    
    # Handle timestamp column if present (set as index for slicing convenience)
    # But keep it as a column for TFT which expects 'timestamp' usually or we handle it
    working_df = df.copy()
    if 'timestamp' in working_df.columns:
        working_df['timestamp'] = pd.to_datetime(working_df['timestamp'])
        working_df = working_df.set_index('timestamp')
    elif not isinstance(working_df.index, pd.DatetimeIndex):
        # Try to find a date column
        date_col = None
        for c in working_df.columns:
            if 'date' in c.lower() or 'time' in c.lower():
                date_col = c
                break
        if date_col:
            working_df[date_col] = pd.to_datetime(working_df[date_col])
            working_df = working_df.set_index(date_col)
        else:
            raise ValueError("DataFrame must have a 'timestamp' column or a DatetimeIndex.")
            
    # Ensure dataframe has a frequency
    freq_name = "steps"
    if working_df.index.freq is None:
        try:
            inferred_freq = pd.infer_freq(working_df.index)
            if inferred_freq:
                working_df = working_df.asfreq(inferred_freq)
                inferred_freq_upper = inferred_freq.upper()
                if 'D' in inferred_freq_upper: freq_name = "days"
                elif 'H' in inferred_freq_upper: freq_name = "hours"
                elif 'M' in inferred_freq_upper: freq_name = "months"
                elif 'W' in inferred_freq_upper: freq_name = "weeks"
            else:
                # Fallback heuristic
                diff = working_df.index.to_series().diff().mode()[0]
                if diff >= pd.Timedelta(days=28): freq_name = "months"
                elif diff >= pd.Timedelta(days=7): freq_name = "weeks"
                elif diff >= pd.Timedelta(days=1): freq_name = "days"
                elif diff >= pd.Timedelta(hours=1): freq_name = "hours"
        except Exception as e:
            print(f"Warning: Error inferring frequency: {e}")
    else:
        freq_str = str(working_df.index.freq).upper()
        if 'D' in freq_str: freq_name = "days"
        elif 'H' in freq_str: freq_name = "hours"
        elif 'M' in freq_str: freq_name = "months"
        elif 'W' in freq_str: freq_name = "weeks"

    # Identify covariates
    if future_cov_cols is None:
        # Default: All non-target columns are future covariates
        # We must exclude non-numeric columns (like ID) as the model expects float inputs
        future_cov_cols = []
        for col in working_df.columns:
            if col == target_col:
                continue
            if col in (past_cov_cols or []):
                continue
            # Exclude id_column explicitly
            if col == 'id_column':
                continue
            # Check if numeric
            if pd.api.types.is_numeric_dtype(working_df[col]):
                future_cov_cols.append(col)
    
    if past_cov_cols is None:
        past_cov_cols = []

    # Infer start date and stride
    stride = forecast_horizon
    total_test_points = num_windows * forecast_horizon
    
    if len(working_df) <= total_test_points + model.input_len:
         # We need at least input_len history before the first test point
        raise ValueError(f"Dataset length ({len(working_df)}) is too small for {num_windows} windows with horizon {forecast_horizon} and input length {model.input_len}.")
        
    start_idx = len(working_df) - total_test_points
    start_date = working_df.index[start_idx]
    end_date = working_df.index[-1]
    
    current_date = start_date
    window = 0
    
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION: {start_date.date()} to {end_date.date()}")
    print(f"Forecast Horizon: {forecast_horizon} {freq_name} | Windows: {num_windows}")
    print(f"{'='*70}\n")
    
    id_column_name = 'series_1' # Dummy ID for result format

    for i in range(num_windows):
        window += 1
        
        # 1. Define Test Range
        if working_df.index.freq:
            test_end = current_date + (working_df.index.freq * (forecast_horizon - 1))
        else:
            # Fallback if no freq object but we have inferred unit? 
            # Just use iloc logic if freq is missing to be safe?
            # Let's try timedelta if possible
            try:
                test_end = current_date + pd.Timedelta(days=forecast_horizon - 1) # Defaulting to days if unknown
            except:
                 # Fallback to integer indexing if date math fails
                 current_loc = working_df.index.get_loc(current_date)
                 test_end = working_df.index[min(current_loc + forecast_horizon - 1, len(working_df)-1)]

        # 2. Prepare Data
        # Train: Up to current_date (exclusive)
        # But for TFT predict, we need history + future horizon.
        # So we need a dataframe that goes up to test_end.
        
        # Training Data: strictly BEFORE current_date
        train_df = working_df[working_df.index < current_date].copy()
        train_len = len(train_df)
        
        # Prediction Input: History (last input_len) + Future (forecast_horizon)
        # We take data up to test_end
        pred_input_df = working_df[working_df.index <= test_end].copy()
        
        # Ground Truth
        test_df = working_df[(working_df.index >= current_date) & (working_df.index <= test_end)].copy()
        
        if len(test_df) < forecast_horizon:
            print(f"Window {window}: Skipping - insufficient test data (Length: {len(test_df)})")
            break
            
        # 3. Train/Update Model
        # We fit on the available training history
        try:
            model.fit(
                train_df.reset_index(),
                target_col=target_col,
                past_cov_cols=past_cov_cols,
                future_cov_cols=future_cov_cols,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose
            )
            
            # 4. Predict
            # pred_input_df needs to be formatted correctly.
            # predict expects a DF with history + future rows.
            forecast = model.predict(pred_input_df.reset_index())
            
            # Extract median prediction (q50)
            predictions = forecast['q50'].values
            
            # 5. Metrics
            actuals = test_df[target_col].values
            
            # Truncate predictions if they exceed actuals (shouldn't happen with correct logic)
            predictions = predictions[:len(actuals)]
            
            # Inverse Transform if scaler provided
            if scaler:
                # Reshape for scaler
                predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
                if np.isinf(mape) or np.isnan(mape):
                    mape = 0.0
            
            rmse = np.sqrt(np.mean((actuals - predictions)**2))
            mae = np.mean(np.abs(actuals - predictions))
            
            metrics['rmse'].append(rmse)
            metrics['mae'].append(mae)
            metrics['mape'].append(mape)
            
            print(f"Window {window:3d} | Date: {current_date.date()} | Train: {train_len:4d} {freq_name} | "
                  f"RMSE: {rmse:6.2f} | MAE: {mae:6.2f} | MAPE: {mape:5.2f}%")
            
            # Store results
            result_df = pd.DataFrame({
                'id': [id_column_name] * len(predictions),
                'timestamp': test_df.index,
                'target_name': target_col,
                'predictions': predictions
            })
            results.append(result_df)

        except Exception as e:
            print(f"Window {window}: Error - {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Move to next fold
        if working_df.index.freq:
             current_date += (working_df.index.freq * stride)
        else:
             current_date += pd.Timedelta(days=stride)
             
        gc.collect()
        keras.backend.clear_session() # Optional: clear session to free memory if re-building

    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Windows: {window}")
    if metrics['rmse']:
        print(f"Overall RMSE: {np.mean(metrics['rmse']):.2f}")
        print(f"Overall MAE:  {np.mean(metrics['mae']):.2f}")
        print(f"Overall MAPE: {np.mean(metrics['mape']):.2f}%")
    print(f"{'='*70}\n")

    if not results:
        return pd.DataFrame()
        
    return pd.concat(results, ignore_index=True)
