import pandas as pd
import matplotlib.pyplot as plt
import holidays
import numpy as np
from typing import List, Optional, Union, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_time_series(
    file_path: str,
    forecast_horizon: int,
    is_holiday: bool = False,
    country: str = None,
    is_weekend: bool = False,
    include_friday_in_weekend: bool = False,
    static_covariates: Optional[str] = None,
    cutoff_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Preprocesses time series data for TFT model.

    Args:
        file_path (str): Path to the data file (.csv or .xlsx).
        forecast_horizon (int): Number of steps to forecast into the future.
        is_holiday (bool): Whether to add a holiday indicator.
        country (str): Country code or name for holidays (e.g., 'Greece', 'US'). Required if is_holiday is True.
        is_weekend (bool): Whether to add a weekend indicator.
        include_friday_in_weekend (bool): If True, Friday is considered a weekend.
        static_covariates (str, optional): Name of the column to use as static covariate (ID). 
                                           It will be renamed to 'id_column'.

    Returns:
        pd.DataFrame: 'history' dataframe with columns [id_column, timestamp, y, ...covariates].
        pd.DataFrame: 'pred_input' dataframe (history + future) for inference.
        pd.DataFrame: 'test_df' dataframe containing actual future values (scaled).
        MinMaxScaler: Scaler used to scale the target column 'y'.
    """
    # 1. Load Data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx")

    # 2. Identify Columns
    # Assumption: 1st column is Date/Timestamp, 2nd is Target (y) if not named explicitly
    # We rename them to standard 'timestamp' and 'y'
    cols = df.columns.tolist()
    
    # Try to find date column
    date_col = None
    for c in cols:
        if 'date' in c.lower() or 'time' in c.lower():
            date_col = c
            break
    if not date_col:
        date_col = cols[0] # Fallback to first column
        
    # Try to find target column
    target_col = None
    # If we found date_col, assume next one is target if not specified?
    # The prompt implies generic, but let's assume the remaining one or 'y' / 'target'
    # Exclude static_covariates from possible targets if provided
    possible_targets = [c for c in cols if c != date_col and c != static_covariates]
    
    # Heuristics for target column name
    common_target_names = ['y', 'target', 'sales', 'traffic', 'demand', 'attendance', 'weekly_sales']
    
    # 1. Check for exact match in common names
    for name in common_target_names:
        for col in possible_targets:
            if col.lower() == name:
                target_col = col
                break
        if target_col:
            break
            
    # 2. If not found, check if only one possible target remains
    if not target_col:
        if len(possible_targets) == 1:
            target_col = possible_targets[0]
        else:
            # 3. Fallback: Pick the first possible target that is NOT the date column (already filtered)
            # and preferably not 'id' or 'store' if possible, but we filtered static_covariates.
            # If we have multiple, we default to the first one.
            if possible_targets:
                target_col = possible_targets[0]
            else:
                # If absolutely no candidates, fallback to 2nd column if it's not date_col, else 1st
                # This is a last resort and might still fail if everything is filtered out.
                if len(cols) > 1 and cols[1] != date_col:
                    target_col = cols[1]
                elif cols[0] != date_col:
                    target_col = cols[0]
                else:
                    raise ValueError("Could not identify target column. Please rename target to 'y' or 'target'.")

    df = df.rename(columns={date_col: 'timestamp', target_col: 'y'})
    
    # Ensure y is float
    df['y'] = df['y'].astype(float)
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Split into train and test if cutoff_date is provided
    if cutoff_date:
        train_df = df[df['timestamp'] <= cutoff_date].copy()
        test_df = df[df['timestamp'] > cutoff_date].copy()
    else:
        # If no cutoff, use all data for fitting (not recommended for strict evaluation)
        train_df = df.copy()
        test_df = pd.DataFrame() # Empty
    
    # Scale target column using training stats
    scaler = MinMaxScaler()
    # Fit on training data
    scaler.fit(train_df[['y']])
    
    # Transform both
    df['y'] = scaler.transform(df[['y']])
    # We also need to update train_df/test_df if we use them later, but we are modifying df in place/reassigning
    # Actually, we should be careful. We just updated df['y'].
    # If we need to scale other covariates, we should do it similarly.
    
    # Infer Frequency
    freq = pd.infer_freq(df['timestamp'])
    if not freq:
        # Simple heuristic if infer_freq fails (e.g. missing data)
        diff = df['timestamp'].diff().mode()[0]
        if diff == pd.Timedelta(days=1):
            freq = 'D'
        # Add more heuristics if needed, or default to 'D'
        else:
            freq = 'D' 

    # 3. Infer Covariates & ID Column
    # Identify ID column
    if static_covariates:
        if static_covariates not in df.columns:
             # It might have been renamed if it was the target or date, but we excluded it from targets.
             # If it was the date column, we have a problem, but date is usually timestamp.
             # Assuming static_covariates is a distinct column.
             if static_covariates == date_col:
                 # If user said date is static covariate, that's weird, but it's now 'timestamp'
                 df['id_column'] = df['timestamp'].astype(str)
             elif static_covariates == target_col:
                 # If user said target is static covariate, that's also weird, but it's now 'y'
                 df['id_column'] = df['y'].astype(str)
             else:
                 raise ValueError(f"Static covariate column '{static_covariates}' not found in dataframe.")
        else:
            df = df.rename(columns={static_covariates: 'id_column'})
            # Ensure it's string
            df['id_column'] = df['id_column'].astype(str)
    elif 'id_column' not in df.columns:
        df['id_column'] = "0" # Default ID as string
        
    # Identify other covariates (exclude timestamp, y, id_column)
    # We assume remaining columns are covariates
    exclude_cols = ['timestamp', 'y', 'id_column']
    potential_covariates = [c for c in df.columns if c not in exclude_cols]
    
    continuous_covariates = []
    categorical_covariates = [] # Including one-hot encoded
    
    for col in potential_covariates:
        # Check if column is one-hot encoded (binary 0/1)
        if df[col].dropna().isin([0, 1]).all():
             categorical_covariates.append(col)
        # Check if numeric
        elif pd.api.types.is_numeric_dtype(df[col]):
             continuous_covariates.append(col)
        else:
             # Treat non-numeric as categorical (though TFT might need encoding, we leave as is for now)
             categorical_covariates.append(col)

    # Normalize continuous covariates using training stats
    # We need to fit a scaler for EACH covariate or use one scaler for all if they are similar?
    # Usually separate scalers or one StandardScaler for the matrix.
    # But we are returning only ONE scaler (for y?).
    # The user said "Validation/testing data should be normalized with the training data stats (mean and std)!"
    # And "return data, pred_input, scaler".
    # If we scale covariates, we should probably keep their scalers or just apply the transformation.
    # Since we only return one scaler, I assume it's primarily for the target 'y' inverse transform.
    # For covariates, we just transform them in the dataframe and don't return their scalers (unless requested).
    
    for col in continuous_covariates:
        # Fit on training part
        if cutoff_date:
            train_vals = df[df['timestamp'] <= cutoff_date][col]
        else:
            train_vals = df[col]
            
        mean = train_vals.mean()
        std = train_vals.std()
        
        if std != 0:
            df[col] = (df[col] - mean) / std
            
    # 4. Time Covariates
    time_covariates = []
    
    if is_holiday:
        if not country:
            raise ValueError("Country must be specified when is_holiday is True.")
        try:
            # Use getattr to fetch country class dynamically or country_holidays if available
            if hasattr(holidays, 'country_holidays'):
                country_holidays = holidays.country_holidays(country)
            else:
                # Fallback for older versions if country_holidays not present (though dir showed it is)
                country_holidays = getattr(holidays, country)()
        except Exception as e:
             # Fallback or re-raise if country is invalid
             raise ValueError(f"Invalid country '{country}' for holidays: {e}")
             
        df['is_holiday'] = df['timestamp'].apply(lambda x: 1 if x in country_holidays else 0)
        time_covariates.append('is_holiday')
        
    if is_weekend:
        weekend_days = [4, 5, 6] if include_friday_in_weekend else [5, 6]
        df['is_weekend'] = df['timestamp'].dt.dayofweek.isin(weekend_days).astype(int)
        time_covariates.append('is_weekend')

    # 5. Future Dataframe (Known Covariates)
    # We use the data after cutoff_date as the future known covariates
    
    if not cutoff_date:
        raise ValueError("cutoff_date is required to split data into history and future for known covariates.")
        
    # Select final columns
    final_cols = ['id_column', 'timestamp', 'y'] + continuous_covariates + categorical_covariates + time_covariates
    df_final = df[final_cols].copy()
    
    # Split into history and future
    data = df_final[df_final['timestamp'] <= cutoff_date].copy()
    future_data = df_final[df_final['timestamp'] > cutoff_date].copy()
    
    if future_data.empty:
        raise ValueError("No data found after cutoff_date. Cannot create future inputs.")
        
    # Use all available future data (known covariates)
    # future_data = future_data.iloc[:forecast_horizon].copy() # Removed slicing
    
    # Mask target in future (avoid leakage)
    future_data['y'] = 0
    
    # Create pred_input
    pred_input = pd.concat([data, future_data], axis=0, ignore_index=True)
    
    # Return data (train), pred_input (train + masked test), test_df (actual test with y), and scaler
    # We need to recreate test_df from df_final split because future_data was modified (y=0)
    # Actually, we can just use the slice from df_final again
    test_df = df_final[df_final['timestamp'] > cutoff_date].copy()
    # Apply scaling to test_df y as well? 
    # Yes, df was scaled in place at line 121: df['y'] = scaler.transform(df[['y']])
    # But wait, df_final was created from df at line 234.
    # df['y'] was scaled at line 121.
    # So df_final has scaled y.
    # So test_df has scaled y.
    
    return data, pred_input, test_df, scaler


def plot_probabilistic_forecast(
    history_df: pd.DataFrame, 
    forecast_df: pd.DataFrame,
    target_col: str = 'y',
    timeseries_name: str = 'Time Series',
    time_col: str = 'timestamp',
    history_length: int = 60,
    scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None,
    actual_df: Optional[pd.DataFrame] = None
):
    """
    Plots the probabilistic forecast along with historical data and optional actuals.
    
    If a scaler is provided, the historical data, forecast, and actuals are inverse transformed to the original scale.
    
    Args:
        history_df (pd.DataFrame): Historical dataframe containing the target column.
        forecast_df (pd.DataFrame): Forecast dataframe with 'q10', 'q50', 'q90' columns.
        target_col (str, optional): Name of target column. Defaults to 'y'.
        timeseries_name (str, optional): Name of the time series for the title. Defaults to 'Time Series'.
        time_col (str, optional): Name of the time column. Defaults to 'timestamp'.
        history_length (int, optional): Number of historical steps to plot. Defaults to 60.
        scaler (Union[MinMaxScaler, StandardScaler], optional): Scaler used to scale the target column. 
                                         If provided, data will be inverse transformed.
        actual_df (pd.DataFrame, optional): Dataframe containing actual values for the forecast period.
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate forecast horizon from forecast dataframe
    forecast_horizon = len(forecast_df)
    x_forecast = None
    
    # Slice History    
    recent_history = history_df.iloc[-history_length:].copy()
    
    # Inverse transform history if scaler provided
    # Inverse transform history if scaler provided
    if scaler:
        recent_history[target_col] = scaler.inverse_transform(recent_history[[target_col]])
        
        # Inverse transform forecast
        # We need to reshape to (-1, 1) for inverse_transform
        if 'q50' in forecast_df.columns:
            forecast_df['q50'] = scaler.inverse_transform(forecast_df['q50'].values.reshape(-1, 1))
        if 'q10' in forecast_df.columns:
            forecast_df['q10'] = scaler.inverse_transform(forecast_df['q10'].values.reshape(-1, 1))
        if 'q90' in forecast_df.columns:
            forecast_df['q90'] = scaler.inverse_transform(forecast_df['q90'].values.reshape(-1, 1))
            
    # Prepare Actuals if provided
    y_actual = None
    if actual_df is not None:
        actuals = actual_df.copy()
        if scaler:
             actuals[target_col] = scaler.inverse_transform(actuals[[target_col]])
        y_actual = actuals[target_col]
    
    # Infer frequency from the dataframe
    freq_unit = "days" # Default
    
    # Ensure time column is datetime
    if time_col in recent_history.columns:
        recent_history[time_col] = pd.to_datetime(recent_history[time_col])
        x_history = recent_history[time_col]
        last_date = recent_history[time_col].iloc[-1]
        freq = pd.infer_freq(x_history)
    else:
        # Fallback if time_col not found or not provided
        recent_history['timestamp'] = pd.to_datetime(recent_history['timestamp'])
        x_history = recent_history['timestamp']
        last_date = recent_history['timestamp'].iloc[-1]
        freq = pd.infer_freq(x_history)
    
    # Infer frequency unit for title
    if freq:
        freq_upper = freq.upper()
        if 'H' in freq_upper: freq_unit = "hours"
        elif 'D' in freq_upper: freq_unit = "days"
        elif 'W' in freq_upper: freq_unit = "weeks"
        elif 'M' in freq_upper: freq_unit = "months"
    else:
         # Fallback heuristic if infer_freq fails
        diff = x_history.diff().mode()[0]
        if diff >= pd.Timedelta(days=28): freq_unit = "months"
        elif diff >= pd.Timedelta(days=7): freq_unit = "weeks"
        elif diff >= pd.Timedelta(days=1): freq_unit = "days"
        elif diff >= pd.Timedelta(hours=1): freq_unit = "hours"
        freq = 'D' # Default for date_range generation if inference failed

    if freq:
        offset = pd.tseries.frequencies.to_offset(freq)
        start_date = last_date + offset
    else:
        start_date = last_date + pd.Timedelta(days=1)
        
    if actual_df is not None and time_col in actual_df.columns:
        # Use actual timestamps if available and align with forecast length
        # We assume actual_df starts at the same time as forecast (immediately after history)
        # But actual_df passed might be the whole test set.
        # We should take the first 'forecast_horizon' rows of actual_df
        if len(actual_df) >= forecast_horizon:
             x_forecast = pd.to_datetime(actual_df[time_col].iloc[:forecast_horizon])
             # Also slice y_actual to match
             if y_actual is not None:
                 y_actual = y_actual.iloc[:forecast_horizon]
        else:
             # Actuals shorter than forecast? Use what we have, and generate the rest?
             # Or just use generated dates.
             # User said "plot series until the prediction ... not further".
             # Let's stick to generated dates but if actuals are available use them to verify/overwrite?
             # Actually, if we have actuals, we prefer their timestamps.
             pass
    
    if x_forecast is None or len(x_forecast) != forecast_horizon:
         # Fallback to generation
         future_dates = pd.date_range(
            start=start_date,
            periods=forecast_horizon,
            freq=freq
        )
         x_forecast = future_dates
         
    # Final safety check: ensure x_forecast and y_actual are sliced to forecast_horizon
    # This handles cases where slicing might have failed or dimensions are off
    if len(x_forecast) > forecast_horizon:
        x_forecast = x_forecast[:forecast_horizon]
        
    if y_actual is not None and len(y_actual) > forecast_horizon:
        y_actual = y_actual[:forecast_horizon]

    y_history = recent_history[target_col]

    # Plot historical data
    plt.plot(x_history, y_history, label='History', color='black', alpha=0.6)
    
    # Connector line
    # Handle x_forecast indexing (could be Series or DatetimeIndex)
    first_forecast_date = x_forecast.iloc[0] if hasattr(x_forecast, 'iloc') else x_forecast[0]
    
    plt.plot(
        [x_history.iloc[-1], first_forecast_date], 
        [y_history.iloc[-1], forecast_df["q50"].iloc[0]], 
        color="#1f77b4", linestyle="--", alpha=0.5
    )
    
    # Median
    plt.plot(x_forecast, forecast_df["q50"], label="Forecast (Median)", color="#1f77b4", linewidth=2)

    # Interval
    if "q10" in forecast_df.columns and "q90" in forecast_df.columns:
        plt.fill_between(
            x_forecast, forecast_df["q10"], forecast_df["q90"], 
            color="#1f77b4", alpha=0.2, label="Confidence (q10-q90)"
        )
        
    # Plot Actuals
    if y_actual is not None:
        # y_actual and x_forecast are ensured to be of length forecast_horizon earlier
        if len(y_actual) == len(x_forecast):
             plt.plot(x_forecast, y_actual, label="Actual", color="green", linestyle="--", alpha=0.8)
        else:
             print(f"Warning: Actuals length ({len(y_actual)}) does not match forecast horizon ({len(x_forecast)}). Plotting actuals without time alignment.")
             plt.plot(y_actual.values, label="Actual", color="green", linestyle="--", alpha=0.8)

    title = f"{timeseries_name} Probabilistic Forecast: Next {forecast_horizon} {freq_unit}"
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(timeseries_name)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_importance(past_imp, future_imp):
    """Plots feature importance for past and future inputs in TFT model."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Past
    ax1.barh(past_imp["Feature"], past_imp["Importance"], color="#1f77b4")
    ax1.set_title("Past Input Importance (Encoder)")
    ax1.set_xlabel("Attention Weight (0-1)")
    ax1.invert_yaxis() # Highest importance on top

    # Plot Future
    ax2.barh(future_imp["Feature"], future_imp["Importance"], color="#ff7f0e")
    ax2.set_title("Future Input Importance (Decoder)")
    ax2.set_xlabel("Attention Weight (0-1)")
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.show()