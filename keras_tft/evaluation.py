import pandas as pd
import numpy as np
import keras
from typing import List, Optional, Union
from .model import TFTForecaster
from .utils import infer_frequency
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def timeseries_cv(
    model: TFTForecaster, 
    df: pd.DataFrame, 
    num_windows: int,
    forecast_horizon: int = 7,
    target_col: str = 'y',
    past_cov_cols: Optional[List[str]] = None,
    future_cov_cols: Optional[List[str]] = None,
    static_cov_cols: Optional[List[str]] = None,
    exogenous: Optional[List[str]] = None,
    epochs: int = 10,
    batch_size: int = 32,
    verbose: int = 0,
    scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None
):
    """
    Perform time series cross-validation with TFT and covariates.
    
    This function automatically detects covariates from the dataframe if not explicitly provided,
    handles different time frequencies by inferring from the dataframe index, and prints detailed 
    performance metrics (RMSE, MAE, MAPE).
    
    The number of windows is fixed, and the start date is inferred to cover exactly
    `num_windows * forecast_horizon` steps at the end of the dataset.
    Stride is set equal to `forecast_horizon` (non-overlapping test sets).

    Args:
        model (TFTForecaster): Initialized TFTForecaster model.
        df (pd.DataFrame): DataFrame with DatetimeIndex or 'timestamp' column. 
                           Must contain the target column.
        num_windows (int): Number of cross-validation windows to perform.
        forecast_horizon (int, optional): Number of steps to forecast ahead. Defaults to 7.
        target_col (str, optional): Name of target column. Defaults to 'y'.
        past_cov_cols (List[str], optional): List of past covariate columns. Defaults to None.
        future_cov_cols (List[str], optional): List of future covariate columns. 
                                               If None, defaults to all non-target columns.
        static_cov_cols (List[str], optional): List of static covariate columns. Defaults to None.
        exogenous (List[str], optional): Alias for future_cov_cols. Defaults to None.
        epochs (int, optional): Number of training epochs per window. Defaults to 10.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        verbose (int, optional): Verbosity mode. Defaults to 0.
        scaler (Union[MinMaxScaler, StandardScaler], optional): Scaler used to inverse transform the target. 
                                                                If None, predictions remain scaled. Defaults to None.
    
    Returns:
        List[pd.DataFrame]: A list of DataFrames, one for each cross-validation window. 
                            Each DataFrame contains:
                            - 'id': The series identifier (if available/inferred).
                            - 'timestamp': The forecast date.
                            - 'target_name': Name of the target variable.
                            - 'predictions': The median (q50) forecast.
                            - 'qXX': Columns for each predicted quantile (e.g., 'q10', 'q90').
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
            
    # Handle Panel Data: Index might have duplicates (multiple series)
    # We need unique timestamps to calculate windows
    unique_timestamps = working_df.index.unique().sort_values()

    # Ensure dataframe has a frequency
    freq_name = "steps"
    if working_df.index.freq is None:
        try:
            inferred_freq = pd.infer_freq(unique_timestamps)
            if inferred_freq:
                # Do NOT use asfreq on working_df if it has duplicates (panel data)
                # working_df = working_df.asfreq(inferred_freq) 
                inferred_freq_upper = inferred_freq.upper()
                if 'D' in inferred_freq_upper: freq_name = "days"
                elif 'H' in inferred_freq_upper: freq_name = "hours"
                elif 'M' in inferred_freq_upper: freq_name = "months"
                elif 'W' in inferred_freq_upper: freq_name = "weeks"
            else:
                # Fallback heuristic using unique timestamps
                diff = unique_timestamps.to_series().diff().mode()[0]
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
        
    time_col = working_df.index.name or 'timestamp'

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
            # Exclude static cols
            if col in (static_cov_cols or []):
                continue
            # Check if numeric
            if pd.api.types.is_numeric_dtype(working_df[col]):
                future_cov_cols.append(col)
    
    if past_cov_cols is None:
        past_cov_cols = []

    # Define id_col for later use
    id_col = static_cov_cols[0] if static_cov_cols else None

    # Validation: Ensure each series has enough history (for Panel Data)
    if static_cov_cols:
        primary_id = static_cov_cols[0]
        if primary_id in working_df.columns:
            unique_ids = working_df[primary_id].unique()
            min_length = model.input_len + forecast_horizon
            for series_id in unique_ids:
                series_df = working_df[working_df[primary_id] == series_id]
                if len(series_df) < min_length:
                    # User requested Error
                    raise ValueError(f"Series {series_id} (Length: {len(series_df)}) is too short for model inputs. Required: {min_length}")

    # Infer start date and stride
    stride = forecast_horizon
    
    test_end_points = []
    last_date = unique_timestamps[-1]
    
    # Infer frequency from unique timestamps if possible
    freq = None
    if working_df.index.freq:
        freq = pd.tseries.frequencies.to_offset(working_df.index.freq)
    else:
        freq = pd.infer_freq(unique_timestamps)
        if freq:
            freq = pd.tseries.frequencies.to_offset(freq)
    
    for i in range(num_windows):
        # Backwards from last_date
        if freq:
            end_dt = last_date - (freq * (stride * i))
        else:
             # Fallback: use integer position on UNIQUE timestamps
             pos = len(unique_timestamps) - 1 - (stride * i)
             if pos < 0: break
             end_dt = unique_timestamps[pos]
        test_end_points.append(end_dt)
        
    test_end_points = sorted(test_end_points) # Chronological order
    
    if len(test_end_points) < num_windows:
         raise ValueError(f"Dataset too short for {num_windows} windows.")

    # Check history for first window
    first_test_end = test_end_points[0]
    # Start of first test window
    if freq:
        first_test_start = first_test_end - (freq * (forecast_horizon - 1))
    else:
        # Heuristic on unique timestamps
        try:
            pos_end = unique_timestamps.get_loc(first_test_end)
            pos_start = pos_end - forecast_horizon + 1
            if pos_start < 0: raise ValueError("Dataset too short.")
            first_test_start = unique_timestamps[pos_start]
        except KeyError:
             # Should not happen if logic is correct
             raise ValueError(f"Could not locate start date for {first_test_end}")
        
    # We need input_len before first_test_start
    # Check if we have enough data before first_test_start
    # We check the earliest timestamp in the dataset
    if unique_timestamps[0] > first_test_start: # This is rough check
         # Better: check count of unique timestamps before first_test_start
         train_len = len(unique_timestamps[unique_timestamps < first_test_start])
         if train_len < model.input_len:
             raise ValueError(f"Not enough history for first window. Need {model.input_len} steps, have {train_len}.")
        


    print(f"\n{'='*95}")
    print(f"CROSS-VALIDATION: {test_end_points[0].date()} to {test_end_points[-1].date()} (Test Ends)")
    print(f"Forecast Horizon: {forecast_horizon} {freq_name} | Windows: {len(test_end_points)}")
    print(f"{'='*95}\n")
    
    id_column_name = 'series_1' # Dummy ID for result format

    for i, test_end in enumerate(test_end_points):
        window = i + 1
        
        # 1. Define Test Range
        # test_end is already defined
        
        # Calculate current_date (start of test window)
        if freq:
            current_date = test_end - (freq * (forecast_horizon - 1))
        else:
            pos_end = unique_timestamps.get_loc(test_end)
            pos_start = pos_end - forecast_horizon + 1
            current_date = unique_timestamps[pos_start]

        # 2. Prepare Data
        # We perform efficient slicing using .loc
        # Use views where possible, but for fit() we might need a copy if it modifies data inplace (it doesn't usually)
        # However, passing slices is safer if they are contiguous
        
        # Training Data: strictly BEFORE current_date
        train_df = working_df.loc[working_df.index < current_date]
        if not train_df.empty:
             train_len = len(train_df)
        else:
             train_len = 0
             
        # Prediction Input: History (last input_len) + Future (forecast_horizon)
        # We take data up to test_end
        pred_input_df = working_df.loc[working_df.index <= test_end]
        
        # Ground Truth
        test_df = working_df.loc[(working_df.index >= current_date) & (working_df.index <= test_end)]
        
        if len(test_df) < forecast_horizon:
            print(f"Window {window}: Skipping - insufficient test data (Length: {len(test_df)})")
            break
            
        # Integrity Check for Panel Data
        if verbose >= 1 and static_cov_cols:
            primary_id = static_cov_cols[0]
            if primary_id in train_df.columns:
                n_ids = train_df[primary_id].nunique()
                # print(f"Window {window}: Training on {len(train_df)} samples from {n_ids} unique series.")
        
        if verbose > 0:
            print(f"Window {window}: Training on {train_len} samples.")
        
        # 3. Train/Update Model
        # We fit on the available training history
        try:
            model.fit(
                train_df,
                target_col=target_col,
                past_cov_cols=past_cov_cols,
                future_cov_cols=future_cov_cols,
                static_cov_cols=static_cov_cols,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose
            )
            
            # 4. Predict
            # pred_input_df needs to be formatted correctly.
            # predict expects a DF with history + future rows.
            # We pass it with index (timestamp) preserved so we can align results.
            forecast = model.predict(pred_input_df)
            
            # Align Forecast and Ground Truth
            # forecast has index=timestamp, columns=[q10, q50, q90, id_column]
            # test_df has index=timestamp, columns=[target_col, id_column, ...]
            
            # Use 'time_col' which was defined earlier
            forecast_reset = forecast.reset_index(names=time_col)
            test_reset = test_df.reset_index(names=time_col)
            
            # Merge on ID and Timestamp
            # Fix per user feedback: handle None id_col safely
            merge_on = [time_col]
            if id_col:
                # Validate id_col exists in both
                id_missing_forecast = id_col not in forecast_reset.columns
                id_missing_test = id_col not in test_reset.columns
                
                if id_missing_forecast:
                     print(f"Warning: ID column '{id_col}' not found in forecast. Merging on time only.")
                if id_missing_test:
                     print(f"Warning: ID column '{id_col}' not found in test data. Merging on time only.")
                     
                if not id_missing_forecast and not id_missing_test:
                    merge_on.insert(0, id_col)
                else:
                    # Fallback to time only if ID missing in either
                    pass # merge_on already [time_col]
            
            if static_cov_cols:
                missing_in_forecast = [c for c in static_cov_cols if c not in forecast_reset.columns]
                missing_in_test = [c for c in static_cov_cols if c not in test_reset.columns]
                
                if missing_in_forecast or missing_in_test:
                    # Only raise if strictly required? TFT predict should handle it.
                    # But for merging we need IDs.
                    if missing_in_test:
                         pass # Test DF from source should have it
                    if missing_in_forecast:
                         # This implies model.predict didn't return ID columns
                         pass 
                
                merge_on.extend([c for c in static_cov_cols if c not in merge_on])
                # Remove duplicates
                merge_on = list(dict.fromkeys(merge_on))
            
            merged = pd.merge(test_reset, forecast_reset, on=merge_on, how='inner', suffixes=('', '_pred'))
            
            if len(merged) == 0:
                print(f"Window {window}: Warning - No overlap between forecast and test data after merge.")
                continue
            
            # Verify no data loss (optional but good for debugging)
            if len(merged) != len(test_reset):
                # This is expected if forecast only covers a subset (e.g. if some series failed)
                # But we should warn if it's significant
                pass

            # Extract prediction (q50) and actuals for metrics
            predictions = merged['q50'].values
            actuals = merged[target_col].values
            
            # Inverse Scale
            if scaler:
                predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
            
            # Metrics
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
                if np.isinf(mape) or np.isnan(mape): mape = 0.0
            
            rmse = np.sqrt(np.mean((actuals - predictions)**2))
            mae = np.mean(np.abs(actuals - predictions))
            
            metrics['rmse'].append(rmse)
            metrics['mae'].append(mae)
            metrics['mape'].append(mape)
            
            print(f"Window {window:3d} | Date: {current_date.date()} | Train: {train_len:4d} {freq_name} | "
                  f"RMSE: {rmse:6.2f} | MAE: {mae:6.2f} | MAPE: {mape:5.2f}%")
            
            # Store results
            # If id_col exists, use it. Else use dummy.
            if id_col:
                res_ids = merged[id_col]
            else:
                res_ids = [id_column_name] * len(predictions)

            result_df = pd.DataFrame({
                'id': res_ids,
                'timestamp': merged[time_col],
                'target_name': target_col,
                'predictions': predictions
            })
            # Add quantile columns if present
            for q_col in [c for c in merged.columns if c.startswith('q') and c not in result_df.columns]:
                 if q_col in merged.columns:
                      result_df[q_col] = merged[q_col]
                      if scaler:
                           result_df[q_col] = scaler.inverse_transform(result_df[[q_col]]).flatten()

            results.append(result_df)

        except Exception as e:
            print(f"Window {window}: Error - {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        # Move to next fold
        # Move to next fold
        if freq:
             current_date += (freq * stride)
        else:
             # Fallback if no freq
             # We can't easily advance without freq unless we use position, 
             # but current_date is a Timestamp.
             # If we are here, it means we inferred freq failed.
             # We should probably use the unique_timestamps logic again or just add days as fallback
             current_date += pd.Timedelta(days=stride)
             
        # gc.collect() # Removed to avoid NameError
        keras.backend.clear_session() # Optional: clear session to free memory if re-building

    print(f"\n{'='*95}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*95}")
    print(f"Total Windows: {window}")
    if metrics['rmse']:
        print(f"Overall RMSE: {np.mean(metrics['rmse']):.2f}")
        print(f"Overall MAE:  {np.mean(metrics['mae']):.2f}")
        print(f"Overall MAPE: {np.mean(metrics['mape']):.2f}%")
    print(f"{'='*95}\n")
        
    return pd.concat(results, ignore_index=True)
