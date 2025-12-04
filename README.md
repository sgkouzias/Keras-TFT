# Keras TFT: Temporal Fusion Transformer

A Keras 3 implementation of the **Temporal Fusion Transformer (TFT)** for interpretable multi-horizon time series forecasting, based on the paper [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Lim et al.

## Features

- **Multi-Horizon Forecasting**: Predicts multiple future time steps simultaneously using quantile regression.
- **Heterogeneous Data Support**: Handles static covariates, past-observed time-varying inputs, and future-known time-varying inputs.
- **Interpretability**:
    - **Variable Selection Networks (VSN)**: Identifies the most important features for prediction.
    - **Attention Weights**: Visualizes temporal patterns and important time steps.
- **State-of-the-Art Architecture**:
    - **Gated Linear Units (GLU)**: For suppressing unused components.
    - **Gated Residual Networks (GRN)**: For processing inputs with non-linear processing and skip connections.
    - **Static Enrichment**: Integrates static context into temporal processing.
    - **Seq2Seq LSTM**: Captures long-term dependencies.
    - **Multi-Head Attention**: Learns long-range relationships.
- **Keras 3 Compatible**: Built to work with TensorFlow, JAX, and PyTorch backends.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install keras pandas numpy matplotlib scikit-learn
```

## Usage

### 1. Initialization

```python
from keras_tft import TFTForecaster

model = TFTForecaster(
    input_chunk_length=24,       # Lookback window size
    output_chunk_length=12,      # Forecast horizon
    hidden_dim=128,              # Hidden dimension size
    quantiles=[0.1, 0.5, 0.9],   # Quantiles to predict
    dropout_rate=0.1,
    num_heads=4,
    optimizer="adam",
    learning_rate=0.001
)
```

### 2. Training

The `fit` method handles data scaling and windowing automatically.

```python
model.fit(
    df,
    target_col="volume",
    past_cov_cols=["past_feature1", "past_feature2"],
    future_cov_cols=["day_of_week", "is_holiday"],
    static_cov_cols=["store_id", "location_type"],
    epochs=50,
    batch_size=64,
    use_lr_schedule=True,        # Uses ReduceLROnPlateau
    use_early_stopping=True,     # Uses EarlyStopping
    early_stopping_patience=10,
    validation_split=0.2
)
```

### 3. Forecasting

```python
# Predicts quantiles for the future horizon
predictions = model.predict(df)

# predictions shape: (num_samples, output_chunk_length, num_quantiles)
```

### 4. Interpretability

Extract global feature importance scores:

```python
past_imp, fut_imp, static_imp = model.get_feature_importance(df)

print("Past Feature Importance:\n", past_imp)
print("Future Feature Importance:\n", fut_imp)
print("Static Feature Importance:\n", static_imp)
```

### 5. Model Summary

Inspect the underlying Keras model architecture:

```python
model.summary()
```

## Architecture Overview

The implementation follows the official TFT architecture:

1.  **Input Processing**: Features are processed by **Variable Selection Networks (VSN)**, conditioned on static context.
2.  **Static Enrichment**: Static covariates are encoded and used to initialize the LSTM and enrich temporal features.
3.  **Temporal Processing**: A sequence-to-sequence **LSTM** (Encoder-Decoder) captures local patterns.
4.  **Attention**: **Multi-Head Attention** captures long-term dependencies across the lookback window.
5.  **Output**: A **Gated Residual Network** and dense layers produce quantile forecasts.

## Project Structure

- `keras_tft/`: Package source code.
    - `model.py`: Main `TFTForecaster` class.
    - `layers.py`: Custom layers (GLU, GRN, VSN, etc.).
    - `loss.py`: Quantile loss function.
    - `utils.py`: Helper functions for plotting and preprocessing.
- `Sales volume prediction with covariates.ipynb`: Example notebook for sales forecasting.
- `Traffic volume prediction with time covariates.ipynb`: Example notebook for traffic forecasting.

## License

[Apache License](LICENSE)
