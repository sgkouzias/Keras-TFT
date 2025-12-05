import keras
from keras import layers, ops
try:
    from keras.preprocessing import timeseries_dataset_from_array
except ImportError:
    from tensorflow.keras.preprocessing import timeseries_dataset_from_array

import tensorflow as tf
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import pandas as pd
from .layers import GatedResidualNetwork, MultivariateVariableSelection, GatedLinearUnit, StaticVariableSelection, InterpretableMultiHeadAttention
from .loss import QuantileLoss

class TFTForecaster:
    """
    Temporal Fusion Transformer (TFT) Forecaster.

    This class implements the TFT architecture for time series forecasting, supporting
    multi-horizon forecasting, static covariates, and interpretable attention mechanisms.
    It wraps the Keras model building, training, and prediction logic.

    Attributes:
        input_len (int): Length of the input sequence (lookback window).
        output_len (int): Length of the output sequence (forecast horizon).
        quantiles (List[float]): List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
        hidden_dim (int): Hidden dimension size for internal layers.
        dropout_rate (float): Dropout rate for regularization.
        num_heads (int): Number of attention heads.
        optimizer_name (str): Name of the optimizer to use.
        learning_rate (float): Learning rate for the optimizer.
        num_past_features (int): Number of past-observed features.
        num_future_features (int): Number of known future features.
        num_static_features (int): Number of static features.
        past_categorical_dict (Dict[int, int]): Dictionary mapping past feature indices to vocab sizes.
        future_categorical_dict (Dict[int, int]): Dictionary mapping future feature indices to vocab sizes.
        static_categorical_dict (Dict[int, int]): Dictionary mapping static feature indices to vocab sizes.
        model (keras.Model): The underlying Keras model.
    """
    def __init__(
        self, 
        input_chunk_length: int, 
        output_chunk_length: int, 
        quantiles: List[float] = [0.1, 0.5, 0.9], 
        hidden_dim: int = 128,
        dropout_rate: float = 0.1,
        num_heads: int = 4,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        # Feature configs required for building at init
        num_past_features: int = 0,
        num_future_features: int = 0,
        num_static_features: int = 0,
        past_categorical_dict: Dict[int, int] = {}, # {idx: vocab_size}
        future_categorical_dict: Dict[int, int] = {},
        static_categorical_dict: Dict[int, int] = {}
    ):
        """
        Initialize the TFTForecaster.

        Args:
            input_chunk_length (int): Number of past time steps to use as input.
            output_chunk_length (int): Number of future time steps to predict.
            quantiles (List[float], optional): Quantiles to predict. Defaults to [0.1, 0.5, 0.9].
            hidden_dim (int, optional): Hidden dimension size. Defaults to 128.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            optimizer (str, optional): Optimizer name. Defaults to "adam".
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            num_past_features (int, optional): Number of past features. Defaults to 0.
            num_future_features (int, optional): Number of future features. Defaults to 0.
            num_static_features (int, optional): Number of static features. Defaults to 0.
            past_categorical_dict (Dict[int, int], optional): Map of past categorical feature indices to vocab sizes. Defaults to {}.
            future_categorical_dict (Dict[int, int], optional): Map of future categorical feature indices to vocab sizes. Defaults to {}.
            static_categorical_dict (Dict[int, int], optional): Map of static categorical feature indices to vocab sizes. Defaults to {}.
        """
        self.input_len = input_chunk_length
        self.output_len = output_chunk_length
        self.quantiles = quantiles
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        
        # Feature configs
        self.num_past_features = num_past_features
        self.num_future_features = num_future_features
        self.num_static_features = num_static_features
        self.past_categorical_dict = past_categorical_dict
        self.future_categorical_dict = future_categorical_dict
        self.static_categorical_dict = static_categorical_dict
        
        self.model = None
        self.explain_model = None
        self.scalers: Dict[str, Tuple[float, float]] = {}
        self.target_col: Optional[str] = None
        self.past_cov_cols: List[str] = []
        self.future_cov_cols: List[str] = []
        self.static_cov_cols: List[str] = []
        self.feature_cols: List[str] = []
        
        # Build model immediately if feature counts are provided (and > 0 implies intent, though 0 is valid for static)
        # We assume if the user initializes with these, they want it built.
        # If they are 0, we might build a degenerate model or wait?
        # The prompt says "The model should be built upon initialization".
        # So we build it.
        self._build_and_compile_model()
        
        # Check integer division for attention heads
        if self.hidden_dim % self.num_heads != 0:
            print(f"Warning: hidden_dim ({self.hidden_dim}) not divisible by num_heads ({self.num_heads}). "
                  f"Key dim will be {self.hidden_dim // self.num_heads}.")
        
        # Validate indices if provided
        # We can't validate fully until we know cols, but if user provided dicts, we assume they know what they are doing OR
        # better validation happens in fit(). here checks are minimal.
        
        # Initial validation call
        if self.static_cov_cols or self.past_cov_cols or self.future_cov_cols: # Only if cols known (usually not at init unless inferred/passed?)
             # Actually cols are not passed to init. So we can't strict validate here unless we assume defaults.
             pass
        # However, we can call it if strict checks are needed later.
        
        # User requested call in __init__ or fit. It's in fit.
        # But let's add it here if consistent.
        # Problem: self.static_cov_cols is empty at init.
        # So _validate_categorical_indices() checks len(...) which is 0.
        # If user passed {0: 10} and len is 0, idx 0 >= 0 -> Raises Error.
        # So we can ONLY call it if feature counts match?
        # But num_static_features is passed.
        # _validate uses len(self.static_cov_cols).
        # At init, static_cov_cols is []. Length 0.
        # So calling it here WILL fail if dict is not empty.
        # So we CANNOT call it in __init__ unless we use num_features instead of col lists.
        # Let's Skip __init__ call and rely on fit(), but ensure fit() call is robust.
    def _build_and_compile_model(self):
        if self.model is not None:
             return

        num_past_features = self.num_past_features
        num_future_features = self.num_future_features
        num_static_features = self.num_static_features
        
        # --- Inputs ---
        input_past = keras.Input(shape=(self.input_len, num_past_features), name="past_input")
        input_future = keras.Input(shape=(self.output_len, num_future_features), name="future_input")
        
        inputs = [input_past, input_future]
        
        # --- Static Covariate Encoders ---
        if num_static_features > 0:
            input_static = keras.Input(shape=(num_static_features,), name="static_input")
            inputs.append(input_static)
            
            # Static VSN (2D)
            static_embedding, static_weights = StaticVariableSelection(
                num_static_features, self.hidden_dim, self.dropout_rate, name="vsn_static",
                categorical_indices=list(self.static_categorical_dict.keys()),
                vocab_sizes=list(self.static_categorical_dict.values())
            )(input_static)
            
            # Create 4 context vectors
            c_s = layers.Dense(self.hidden_dim, name="static_selection")(static_embedding)
            c_e = layers.Dense(self.hidden_dim, name="static_enrich")(static_embedding)
            c_h = layers.Dense(self.hidden_dim, name="static_h")(static_embedding)
            c_c = layers.Dense(self.hidden_dim, name="static_c")(static_embedding)
            
        else:
            # Zero context if no static features
            # GlobalAveragePooling1D: (Batch, Time, Feat) -> (Batch, Feat)
            dummy = layers.GlobalAveragePooling1D()(input_past) 
            static_embedding = layers.Dense(self.hidden_dim, kernel_initializer='zeros', bias_initializer='zeros')(dummy)
            # Ensure it's zero
            static_embedding = layers.Lambda(lambda x: x * 0)(static_embedding)
            
            c_s = c_e = c_h = c_c = static_embedding
            static_weights = None # No static weights

        # 1. Variable Selection Networks
        
        # Past VSN
        # Use c_s for context
        if num_past_features > 0:
            x_past, past_weights = MultivariateVariableSelection(
                num_past_features, self.hidden_dim, self.dropout_rate, name="vsn_past",
                categorical_indices=list(self.past_categorical_dict.keys()),
                vocab_sizes=list(self.past_categorical_dict.values())
            )([input_past, c_s])
        else:
            # Dummy zero output
            x_past = layers.Lambda(lambda x: ops.zeros((ops.shape(x)[0], self.input_len, self.hidden_dim)))(input_past)
            past_weights = layers.Lambda(lambda x: ops.zeros((ops.shape(x)[0], 0)))(input_past)
        
        # Future VSN
        # Use c_s for context
        if num_future_features > 0:
            x_fut, future_weights = MultivariateVariableSelection(
                num_future_features, self.hidden_dim, self.dropout_rate, name="vsn_future",
                categorical_indices=list(self.future_categorical_dict.keys()),
                vocab_sizes=list(self.future_categorical_dict.values())
            )([input_future, c_s])
        else:
            # Dummy zero output
            x_fut = layers.Lambda(lambda x: ops.zeros((ops.shape(x)[0], self.output_len, self.hidden_dim)))(input_future)
            future_weights = layers.Lambda(lambda x: ops.zeros((ops.shape(x)[0], 0)))(input_future)

        # 2. LSTM Encoder-Decoder (Seq2Seq)
        # Explicitly create shared LSTM layer for correct weight sharing
        lstm_shared = layers.LSTM(self.hidden_dim, return_sequences=True, return_state=True, name="lstm_shared")
        
        # Static Enrichment for Past (use c_e)
        x_past = GatedResidualNetwork(self.hidden_dim, self.dropout_rate, name="grn_enrich_past")([x_past, c_e])
        
        # Static Enrichment for Future (use c_e)
        x_fut = GatedResidualNetwork(self.hidden_dim, self.dropout_rate, name="grn_enrich_fut")([x_fut, c_e])

        # Concatenate for full sequence LSTM processing
        # This ensures proper state passing from past to future without manual state management quirks
        x_all = ops.concatenate([x_past, x_fut], axis=1)
        
        # Run LSTM on full sequence (Past + Future)
        # Initialize with static context
        all_lstm_out, _, _ = lstm_shared(x_all, initial_state=[c_h, c_c])
        
        # Split back into encoder and decoder parts
        encoder_out_raw = all_lstm_out[:, :self.input_len, :]
        decoder_out_raw = all_lstm_out[:, self.input_len:, :]
        
        # Post-LSTM Gate (GLU) + Add + Norm for Encoder
        encoder_out = GatedLinearUnit(self.hidden_dim, self.dropout_rate)(encoder_out_raw)
        encoder_out = layers.LayerNormalization()(encoder_out + x_past) # Residual from x_past

        # Post-LSTM Gate + Add + Norm for Decoder
        decoder_out = GatedLinearUnit(self.hidden_dim, self.dropout_rate)(decoder_out_raw)
        decoder_out = layers.LayerNormalization()(decoder_out + x_fut) # Residual from x_fut

        # 3. Multi-Head Attention (Interpretable)
        # Returns (output, weights) if return_attention_scores=True
        # Use hidden_dim // num_heads for key_dim to maintain standard MHA dimensionality
        # and ensure concatenation results in hidden_dim (roughly) or at least controllable projection
        attn_layer = InterpretableMultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.hidden_dim // self.num_heads, 
            dropout=self.dropout_rate, output_dim=self.hidden_dim
        )
        attn_out, attn_weights = attn_layer(
            query=decoder_out, value=encoder_out, key=encoder_out, return_attention_scores=True
        )
        
        # Gating for Attention output
        attn_out = GatedLinearUnit(self.hidden_dim, self.dropout_rate)(attn_out)
        attn_out = layers.LayerNormalization()(attn_out + decoder_out) # Residual from decoder_out
        
        # 4. Position-wise Feed Forward (GRN)
        output_grn = GatedResidualNetwork(self.hidden_dim, self.dropout_rate, name="grn_output")
        outputs = output_grn(attn_out)
        
        # Final Gate + Add + Norm
        outputs = GatedLinearUnit(self.hidden_dim, self.dropout_rate)(outputs)
        outputs = layers.LayerNormalization()(outputs + attn_out)

        # 5. Output Head (Quantiles)
        output_dim = len(self.quantiles)
        predictions = layers.Dense(output_dim)(outputs) 
        
        self.model = keras.Model(inputs=inputs, outputs=predictions, name="TemporalFusionTransformer")
        
        # --- Explainability Model ---
        # Outputs: past_weights, future_weights, static_weights (if any), attention_scores
        explain_outputs = {
            "past_weights": past_weights,
            "future_weights": future_weights,
            "attention_scores": attn_weights
        }
        if static_weights is not None:
            explain_outputs['static_weights'] = static_weights
            
        self.explain_model = keras.Model(inputs=inputs, outputs=explain_outputs)
        
        # Configure Optimizer with clipnorm
        if self.optimizer_name.lower() == "adam":
            opt = keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        elif self.optimizer_name.lower() == "rmsprop":
            opt = keras.optimizers.RMSprop(learning_rate=self.learning_rate, clipnorm=1.0)
        elif self.optimizer_name.lower() == "sgd":
            opt = keras.optimizers.SGD(learning_rate=self.learning_rate, clipnorm=1.0)
        else:
            try:
                 opt_cls = getattr(keras.optimizers, self.optimizer_name)
                 opt = opt_cls(learning_rate=self.learning_rate, clipnorm=1.0)
            except:
                 print(f"Warning: Could not instantiate optimizer '{self.optimizer_name}' with learning_rate. Using default.")
                 opt = self.optimizer_name

        self.model.compile(optimizer=opt, loss=QuantileLoss(self.quantiles))


    def get_feature_importance(self, df: pd.DataFrame):
        """
        Extract feature importance using the explainability model.

        Calculates the average attention weights for past, future, and static features
        over a sample of the provided DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to calculate importance from.

        Returns:
            Tuple[pd.DataFrame, ...]: DataFrames containing feature importance scores for 
            past, future, and (optionally) static features.
        
        Raises:
            ValueError: If the model is not fitted or explainability model is unavailable.
        """
        if self.model is None or not hasattr(self, 'explain_model'):
            raise ValueError("Model not fitted or explainability unavailable.")
        
        # Prepare data (limit to 100 samples)
        matrix_df = self._scale_matrix(df, fit=False)
        matrix_vals = matrix_df.values
        
        col_to_idx = {name: i for i, name in enumerate(matrix_df.columns)}
        past_cols = [self.target_col] + self.past_cov_cols + self.future_cov_cols
        fut_cols = self.future_cov_cols
        static_cols = self.static_cov_cols
        
        past_idxs = [col_to_idx[c] for c in past_cols]
        fut_idxs = [col_to_idx[c] for c in fut_cols]
        static_idxs = [col_to_idx[c] for c in static_cols] if static_cols else []
        
        X_past, X_fut, X_static = [], [], []
        num_total = len(matrix_df) - self.input_len - self.output_len
        if num_total <= 0:
             raise ValueError("Dataframe too short for importance calculation.")
             
        num_samples = min(100, num_total)
        step_size = max(1, num_total // num_samples)
        
        for i in range(0, num_total, step_size):
            if len(X_past) >= num_samples: break
            X_past.append(matrix_vals[i:i+self.input_len, past_idxs])
            X_fut.append(matrix_vals[i+self.input_len:i+self.input_len+self.output_len, fut_idxs])
            if static_idxs:
                X_static.append(matrix_vals[i, static_idxs])
        
        inputs = [np.array(X_past), np.array(X_fut)]
        if static_idxs:
            inputs.append(np.array(X_static))
        
        # Get weights from explainability model
        outputs = self.explain_model.predict(inputs, verbose=0)
        
        # Average and format
        # outputs['past_weights'] shape: (Batch, Features) or (Batch, Time, Features)?
        # In MultivariateVariableSelection, we return `weights` which is (Batch, Features) because of temporal averaging.
        # So np.mean(axis=0) is correct.
        
        past_imp = pd.DataFrame({
            "Feature": past_cols, 
            "Importance": np.mean(outputs['past_weights'], axis=0)
        }).sort_values("Importance", ascending=False)
        
        fut_imp = pd.DataFrame({
            "Feature": fut_cols,
            "Importance": np.mean(outputs['future_weights'], axis=0)
        }).sort_values("Importance", ascending=False)
        
        if 'static_weights' in outputs and outputs['static_weights'] is not None:
            static_imp = pd.DataFrame({
                "Feature": static_cols,
                "Importance": np.mean(outputs['static_weights'], axis=0)
            }).sort_values("Importance", ascending=False)
            return past_imp, fut_imp, static_imp
        
        return past_imp, fut_imp

    def _get_categorical_cols(self) -> Set[str]:
        past_cols = list(dict.fromkeys([self.target_col] + self.past_cov_cols + self.future_cov_cols))
        fut_cols = self.future_cov_cols
        static_cols = self.static_cov_cols
        
        categorical_cols = set()
        
        # Static
        for idx in self.static_categorical_dict.keys():
            if idx < len(static_cols):
                categorical_cols.add(static_cols[idx])
                
        # Past
        for idx in self.past_categorical_dict.keys():
            if idx < len(past_cols):
                categorical_cols.add(past_cols[idx])
                
        # Future
        for idx in self.future_categorical_dict.keys():
            if idx < len(fut_cols):
                categorical_cols.add(fut_cols[idx])
                
        return categorical_cols

    def _validate_categorical_indices(self):
        """Validates that configured categorical indices are within bounds of feature arrays."""
        # Static
        static_len = len(self.static_cov_cols)
        for idx in self.static_categorical_dict.keys():
            if idx >= static_len:
                raise ValueError(f"Static categorical index {idx} out of range for {static_len} static features.")

        # Past (includes target + past + future in that order for matrix construction)
        # Note: Past VSN inputs depends on how we construct the array.
        # In this implementation, specific indices are mapped in `call` layers.
        # But `num_past_features` usually implies the width of the past input.
        # Our `_create_tft_dataset` constructs past_input of width `len(past_cols)`.
        past_cols = list(dict.fromkeys([self.target_col] + self.past_cov_cols + self.future_cov_cols))
        past_len = len(past_cols)
        for idx in self.past_categorical_dict.keys():
            if idx >= past_len:
                raise ValueError(f"Past categorical index {idx} out of range for {past_len} past features.")

        # Future
        fut_len = len(self.future_cov_cols)
        for idx in self.future_categorical_dict.keys():
            if idx >= fut_len:
                raise ValueError(f"Future categorical index {idx} out of range for {fut_len} future features.")

    def _validate_categorical_values(self, df: pd.DataFrame):
        """
        Validates that categorical column values in the dataframe are within the configured vocabulary bounds.
        
        Args:
            df (pd.DataFrame): The dataframe containing categorical columns to check.
            
        Raises:
            ValueError: If a value in a categorical column exceeds the size of its corresponding vocabulary.
        """
        # Static
        for idx, vocab in self.static_categorical_dict.items():
            if idx < len(self.static_cov_cols):
                col = self.static_cov_cols[idx]
                if col in df.columns:
                    max_val = df[col].max()
                    if max_val >= vocab:
                        raise ValueError(f"Categorical column '{col}' has value {max_val} >= vocab_size {vocab}")
        
        # Past (includes target + past + future)
        past_cols = list(dict.fromkeys([self.target_col] + self.past_cov_cols + self.future_cov_cols))
        for idx, vocab in self.past_categorical_dict.items():
            if idx < len(past_cols):
                col = past_cols[idx]
                if col in df.columns:
                    max_val = df[col].max()
                    if max_val >= vocab:
                        raise ValueError(f"Categorical column '{col}' has value {max_val} >= vocab_size {vocab}")
        
        # Future
        for idx, vocab in self.future_categorical_dict.items():
            if idx < len(self.future_cov_cols):
                col = self.future_cov_cols[idx]
                if col in df.columns:
                    max_val = df[col].max()
                    if max_val >= vocab:
                        raise ValueError(f"Categorical column '{col}' has value {max_val} >= vocab_size {vocab}")

    def _create_tft_dataset(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> tf.data.Dataset:
        """Creates a TensorFlow Dataset for the TFT model pipeline.
        
        This method constructs a highly optimized `tf.data.Dataset` that handles:
        - Automatic windowing of time series data
        - Grouping by static covariates for panel data
        - Broadcasting of static features to all timesteps in a window
        - Aligning past inputs, future inputs, and target sequences
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe containing all necessary columns 
                (targets, past/future/static covariates) and scaled values.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
            seed (int, optional): Random seed for shuffling reproducibility. Defaults to None.
            
        Returns:
            tf.data.Dataset: A dataset yielding tuples of `(inputs, targets)` where:
                - `inputs`: A tuple/list containing `(past_inputs, future_inputs)` and optionally `static_inputs`.
                - `targets`: The target sequence for the forecast horizon.
                
        Raises:
            ValueError: If the input dataframe does not contain valid time series segments 
                longer than `input_chunk_length + output_chunk_length`.
        """
            
        # Derive column indices
        past_cols = list(dict.fromkeys([self.target_col] + self.past_cov_cols + self.future_cov_cols))
        fut_cols = self.future_cov_cols
        static_cols = self.static_cov_cols
        
        # Optimization: Sort by static columns + timestamp to ensure contiguous memory blocks for groups
        if static_cols:
            sort_cols = static_cols + ['timestamp']
        else:
            sort_cols = ['timestamp']
            
        # Sort dataframe. This creates a copy but ensures optimal access pattern for subsequent steps.
        df_sorted = df.sort_values(sort_cols)
        
        # Scale (returns new DF)
        matrix_df = self._scale_matrix(df_sorted, fit=True, categorical_cols=self._get_categorical_cols())
        
        col_to_idx = {name: i for i, name in enumerate(matrix_df.columns)}
        
        past_idxs = [col_to_idx[c] for c in past_cols]
        fut_idxs = [col_to_idx[c] for c in fut_cols]
        static_idxs = [col_to_idx[c] for c in static_cols] if static_cols else []
        target_idx = col_to_idx[self.target_col]
        
        # Convert to float32 numpy array immediately. 
        # This is the PRIMARY data copy that datasets will view into.
        data = matrix_df.values.astype('float32')
        datasets = []
        
        # Identify group boundaries
        if static_cols:
            # We use groupby size on the SORTED dataframe to get contiguous chunk sizes
            # sort=False maintains order of appearance (which is sorted order due to df_sorted)
            group_sizes = df_sorted.groupby(static_cols, sort=False).size()
            
            start_idx = 0
            for group_key, size in group_sizes.items():
                end_idx = start_idx + size
                
                if size >= self.input_len + self.output_len:
                    # Create VIEW of the data for this group
                    # Since data is contiguous, this slice is a view, avoiding copy
                    group_data_view = data[start_idx:end_idx]
                    
                    self._make_dataset_from_view(datasets, group_data_view, past_idxs, fut_idxs, target_idx, static_idxs)
                
                start_idx = end_idx
        else:
            # Single group
            if len(data) >= self.input_len + self.output_len:
                self._make_dataset_from_view(datasets, data, past_idxs, fut_idxs, target_idx, static_idxs)
        
        if not datasets:
            err_msg = (f"No valid time series found. Required length: {self.input_len + self.output_len}, "
                       f"Dataframe rows: {len(df)}")
            if static_cols:
                # Need variables from closure? No, group_sizes is local.
                # Re-calculate group sizes for error message if we want precise count
                num_groups = df_sorted.groupby(static_cols, sort=False).ngroups
                err_msg += f", Groups: {num_groups}"
            raise ValueError(err_msg)
        
        # Combine all series
        if len(datasets) == 1:
            combined_ds = datasets[0]
        else:
            # Interleave for better series mixing
            # Option 2: Use reduce + concatenate (for sequential chaining)
            # This is safer than from_tensor_slices which doesn't support list of Datasets
            combined_ds = datasets[0]
            for ds in datasets[1:]:
                combined_ds = combined_ds.concatenate(ds)
        
        # Shuffle, batch, prefetch
        if shuffle:
            combined_ds = combined_ds.shuffle(buffer_size=10000, seed=seed)
        
        combined_ds = combined_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return combined_ds

    def _make_dataset_from_view(self, datasets, group_data, past_idxs, fut_idxs, target_idx, static_idxs):
        """Helper to create and append dataset from a data view."""
        total_len = len(group_data)
        
        # Static features (broadcasted)
        if static_idxs:
            static_array = group_data[0, static_idxs] # Already float32
        
        # Past sequences
        past_ds = timeseries_dataset_from_array(
            data=group_data[:total_len - self.output_len, past_idxs],
            targets=None,
            sequence_length=self.input_len,
            sequence_stride=1,
            batch_size=None,
            shuffle=False
        )
        
        # Future sequences
        future_ds = timeseries_dataset_from_array(
            data=group_data[self.input_len:, fut_idxs],
            targets=None,
            sequence_length=self.output_len,
            sequence_stride=1,
            batch_size=None,
            shuffle=False
        )
        
        # Target sequences
        target_ds = timeseries_dataset_from_array(
            data=group_data[self.input_len:, target_idx:target_idx+1],
            targets=None,
            sequence_length=self.output_len,
            sequence_stride=1,
            batch_size=None,
            shuffle=False
        )
        
        # Combine
        if static_idxs:
            series_ds = tf.data.Dataset.zip((past_ds, future_ds, target_ds)).map(
                lambda x_past, x_fut, y: ((x_past, x_fut, tf.constant(static_array)), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            series_ds = tf.data.Dataset.zip((past_ds, future_ds, target_ds)).map(
                lambda x_past, x_fut, y: ((x_past, x_fut), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        datasets.append(series_ds)

    def _scale_matrix(self, df: pd.DataFrame, fit: bool = False, categorical_cols: set = None) -> pd.DataFrame:
        """
        Scales the input DataFrame using standard scaling (mean/std).
        
        Args:
            df (pd.DataFrame): Input DataFrame to scale.
            fit (bool, optional): Whether to fit the scalers on this data. Defaults to False.
            categorical_cols (set, optional): Set of column names to exclude from scaling. Defaults to None.

        Returns:
            pd.DataFrame: Scaled DataFrame.

        Raises:
            ValueError: If a column is not found in fitted scalers (when fit=False) or if a categorical column is not numeric.
        """
        if categorical_cols is None: categorical_cols = set()
        
        data_dict = {}
        for col in self.feature_cols:
            if col in categorical_cols:
                # Do not scale, keep as is (assuming already numeric/encoded)
                # Ensure it's numeric for numpy array conversion later
                # If it's string, we might need LabelEncoder, but for now assume pre-encoded or numeric ID
                try:
                    val = df[col].values.astype(float)
                except ValueError:
                    # If string, try to encode? Or just raise error?
                    # For now, let's assume user provides numeric IDs as per instructions
                    # But if they provide strings, we could map them?
                    # Let's just try to keep as object if conversion fails, but downstream expects float/int array
                    # Actually, TFT expects integer inputs for embeddings.
                    # So we should cast to int?
                    # But _scale_matrix returns a single DF, usually float.
                    # If we mix types, matrix_vals will be object.
                    # This might break numpy slicing if not careful.
                    # Let's assume they are numeric IDs.
                    raise ValueError(f"Categorical column {col} must be numeric (integer IDs).")
                
                data_dict[col] = val
                # We don't add to self.scalers
            else:
                val = df[col].values.astype(float)
                if fit:
                    mean, std = np.mean(val), np.std(val)
                    if std == 0 or np.isnan(std):
                        print(f"Warning: Column '{col}' has zero variance, centering at 0")
                        std = 1.0 
                        # Consistent with utils.py which sets it to 0. 
                        # casting to (val-mean)/1.0 is equivalent to val-mean since val=mean.
                    self.scalers[col] = (mean, std)
                
                if col not in self.scalers:
                     # Strict check for unseen columns
                     raise ValueError(f"Column '{col}' not fitted. Available: {list(self.scalers.keys())}")
                else:
                    mean, std = self.scalers[col]
                
                # Explicit centering for zero variance (std=1.0)
                data_dict[col] = (val - mean) / std
            
        return pd.DataFrame(data_dict, index=df.index)

    def fit(self, df, target_col, past_cov_cols=None, future_cov_cols=None, 
            static_cov_cols=None, exogenous=None, epochs=10, batch_size=32, 
            verbose=1, validation_split=0.0, use_lr_schedule=True,
            use_early_stopping=False, early_stopping_patience=10, seed=None):
        """Trains the TFT model on the provided time series data.

        This method handles data preparation, creating a `tf.data` pipeline, and fitting the Keras model.
        It supports automatic feature detection and model rebuilding if the feature configuration changes.

        Args:
            df (pd.DataFrame): The training dataframe containing the target and all covariates.
                Must have a DatetimeIndex or a column named 'timestamp'.
            target_col (str): The name of the target variable column.
            past_cov_cols (List[str], optional): List of columns to use as past-observed covariates.
                Defaults to None.
            future_cov_cols (List[str], optional): List of columns to use as known future covariates.
                Defaults to None.
            static_cov_cols (List[str], optional): List of columns to use as static covariates
                (e.g., store ID, region). These are used for grouping panel data. Defaults to None.
            exogenous (List[str], optional): Alias for `future_cov_cols` for backward compatibility.
                If provided, it overrides `future_cov_cols` and sets `past_cov_cols` to empty.
                Defaults to None.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            batch_size (int, optional): Batch size. Defaults to 32.
            verbose (int, optional): Verbosity mode. Defaults to 1.
            validation_split (float, optional): Fraction of data to use for validation. Defaults to 0.0.
            use_lr_schedule (bool, optional): Whether to use ReduceLROnPlateau callback. Defaults to True.
            use_early_stopping (bool, optional): Whether to use EarlyStopping callback. Defaults to False.
            early_stopping_patience (int, optional): Patience for early stopping. Defaults to 10.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        
        Note:
            If `fit` is called multiple times with different feature counts, the model will be 
            rebuilt and initialized from scratch. In this case, any existing categorical 
            configuration dictionaries will be cleared to prevent inconsistency, and a warning 
            will be issued.
        """
        # Validate inputs
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe. Available: {list(df.columns)}")
        
        if len(df) < self.input_len + self.output_len:
            raise ValueError(f"Dataframe too short ({len(df)} rows). Need at least {self.input_len + self.output_len}")
        
        # Check for NaN in target
        if df[target_col].isna().any():
            raise ValueError(f"Target column '{target_col}' contains NaN values. Please handle them first.")
            
        # Feature configs must be set before building
        # ...
        
        # Setup column mappings
        self.target_col = target_col
        if exogenous is not None:
            self.future_cov_cols = exogenous
            self.past_cov_cols = []
        else:
            self.past_cov_cols = past_cov_cols if past_cov_cols else []
            self.future_cov_cols = future_cov_cols if future_cov_cols else []
        
        self.static_cov_cols = static_cov_cols if static_cov_cols else []
        self.feature_cols = list(set([target_col] + self.past_cov_cols + 
                                      self.future_cov_cols + self.static_cov_cols))
        
        # Validate categorical indices against feature counts immediately
        self._validate_categorical_indices()
        
        # Derive columns for checks
        past_cols = list(dict.fromkeys([target_col] + self.past_cov_cols + self.future_cov_cols))
        fut_cols = self.future_cov_cols
        static_cols = self.static_cov_cols

        # Check if we need to rebuild due to feature count mismatch
        current_past = self.num_past_features
        current_future = self.num_future_features
        current_static = self.num_static_features
        
        new_past = len(past_cols)
        new_future = len(fut_cols)
        new_static = len(static_cols)
        
        if (current_past != new_past or 
            current_future != new_future or 
            current_static != new_static):

            # Check if we have categorical configs that would be invalidated
            # Only strictly enforce this if the model was already initialized (counts > 0)
            initialized = (current_past > 0 or current_future > 0 or current_static > 0)
            
            if initialized and (self.past_categorical_dict or self.future_categorical_dict or self.static_categorical_dict):
                 if verbose > 0:
                     print("Warning: Clearing categorical dictionaries due to feature count change. Please re-configure if needed.")
                 self.past_categorical_dict = {}
                 self.future_categorical_dict = {}
                 self.static_categorical_dict = {}

            if verbose > 0:
                print(f"WARNING: Feature counts changed. Rebuilding model. "
                      f"Past: {current_past}->{new_past}, Future: {current_future}->{new_future}, Static: {current_static}->{new_static}. "
                      f"This will reset model weights.")
            
            self.num_past_features = new_past
            self.num_future_features = new_future
            self.num_static_features = new_static
            
            self.model = None
            self.explain_model = None
            
            self._build_and_compile_model()
        
        if self.model is None:
             self.num_past_features = len(past_cols)
             self.num_future_features = len(fut_cols)
             self.num_static_features = len(static_cols)
             self._build_and_compile_model()
        
        # Create tf.data pipeline
        dataset = self._create_tft_dataset(
            df=df,
            batch_size=batch_size,
            shuffle=True,
            seed=seed
        )
        
        # Validation split
        val_dataset = None
        if validation_split > 0:
            # Cache the dataset to prevent exhausting the iterator during counting
            dataset = dataset.cache()
            
            # Count total batches safely
            total_batches = dataset.reduce(0, lambda x, _: x + 1).numpy()
            val_batches = int(total_batches * validation_split)
            
            val_dataset = dataset.take(val_batches)
            dataset = dataset.skip(val_batches)
        
        # Callbacks
        callbacks = []
        if use_lr_schedule:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_dataset else 'loss',
                factor=0.5, patience=3, min_lr=1e-6, verbose=verbose
            ))
        
        if use_early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if val_dataset else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            ))
        
        # Train
        if verbose > 0:
            print(f"Training with tf.data pipeline...")
            print(f"Past features: {len(past_cols)} | "
                  f"Future features: {len(fut_cols)} | "
                  f"Static features: {len(static_cols)}")
        
        # Validation: Check categorical max values against vocab bounds
        self._validate_categorical_values(df)
                     
        self.model.fit(
            dataset,
            validation_data=val_dataset,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks
        )

    def summary(self):
        """
        Prints the summary of the underlying Keras model.
        
        If the model has not been built yet, prints a message indicating so.
        """
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet. Call fit() first.")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate probabilistic forecasts for the given DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing historical data and future covariates.
                               Must include the target column (for history) and all covariate columns.
                               For future covariates, values must be provided for the forecast horizon.

        Returns:
            pd.DataFrame: DataFrame containing the forecasts (quantiles) and the timestamp index.
                          Columns: 'q10', 'q50', 'q90' (depending on quantiles), and 'id_column' (if panel data).
        
        Raises:
            ValueError: If the model is not fitted or if input data is insufficient.
        """
        if self.model is None: raise ValueError("Model not fitted.")
        
        # Validate future covariates are present if expected
        if self.future_cov_cols:
            missing = set(self.future_cov_cols) - set(df.columns)
            if missing:
                raise ValueError(f"Future covariates missing in prediction data: {missing}")
                
        # Validate categorical bounds for prediction data
        self._validate_categorical_values(df)

        # Re-derive categorical cols (should refactor this into a method)
        past_cols = list(dict.fromkeys([self.target_col] + self.past_cov_cols + self.future_cov_cols))
        fut_cols = self.future_cov_cols
        static_cols = self.static_cov_cols
        
        categorical_cols = set()
        for idx in self.static_categorical_dict.keys():
            if idx < len(static_cols): categorical_cols.add(static_cols[idx])
        for idx in self.past_categorical_dict.keys():
            if idx < len(past_cols): categorical_cols.add(past_cols[idx])
        for idx in self.future_categorical_dict.keys():
            if idx < len(fut_cols): categorical_cols.add(fut_cols[idx])
                
        # Scale
        matrix_df = self._scale_matrix(df, fit=False, categorical_cols=categorical_cols)
        matrix_vals = matrix_df.values
        col_to_idx = {name: i for i, name in enumerate(matrix_df.columns)}
        
        past_idxs = [col_to_idx[c] for c in past_cols]
        fut_idxs = [col_to_idx[c] for c in fut_cols]
        static_idxs = [col_to_idx[c] for c in static_cols]
        
        # Helper to predict for a single sequence
        def predict_single(seq_vals):
            total_required = self.input_len + self.output_len
            if len(seq_vals) < total_required:
                # Pad with zeros or error?
                # Error is safer.
                raise ValueError(f"Sequence length {len(seq_vals)} < required {total_required}")
                
            # 1. Past: The data BEFORE the forecast horizon
            past_seq = seq_vals[-(self.input_len + self.output_len) : -self.output_len, past_idxs]
            
            # 2. Future: The data DURING the forecast horizon
            fut_seq = seq_vals[-self.output_len:, fut_idxs]
            
            # 3. Static: Take from the last row
            if static_idxs:
                static_seq = seq_vals[-1, static_idxs]
            else:
                static_seq = np.zeros(0)
            
            # Add batch dim
            past_seq = past_seq[np.newaxis, ...]
            fut_seq = fut_seq[np.newaxis, ...]
            
            inputs = [past_seq, fut_seq]
            if static_cols:
                static_seq = static_seq[np.newaxis, ...]
                inputs.append(static_seq)
            
            pred_scaled = self.model.predict(inputs, verbose=0)
            
            # Inverse Scale
            mean, std = self.scalers[self.target_col]
            pred_actual = (pred_scaled * (std + 1e-7)) + mean
            
            return pred_actual[0] # (Output_Len, Quantiles)

        results_list = []
        
        if self.static_cov_cols:
            # Group by original DF to preserve ID types/values
            grouped = df.groupby(self.static_cov_cols)
            for group_name, group_df in grouped:
                if len(group_df) < self.input_len + self.output_len:
                    print(f"Warning: Skipping group {group_name} - insufficient data ({len(group_df)} < {self.input_len + self.output_len})")
                    continue

                # Get scaled values using the index
                scaled_group_vals = matrix_df.loc[group_df.index].values
                pred = predict_single(scaled_group_vals)
                
                # Create result dict for this group
                res = {}
                for i, q in enumerate(self.quantiles):
                    res[f"q{int(q*100):02d}"] = pred[:, i]
                
                res_df = pd.DataFrame(res)
                # Add Timestamp Index (corresponding to the future horizon)
                if 'timestamp' in group_df.columns:
                     res_df.index = group_df['timestamp'].iloc[-self.output_len:]
                else:
                     res_df.index = group_df.index[-self.output_len:]
                
                # Add ID columns
                if isinstance(group_name, tuple):
                    for idx, col in enumerate(self.static_cov_cols):
                        res_df[col] = group_name[idx]
                else:
                    res_df[self.static_cov_cols[0]] = group_name
                    
                results_list.append(res_df)
                
            final_df = pd.concat(results_list, ignore_index=False)
            return final_df
            
        else:
            # Single Series
            pred = predict_single(matrix_vals)
            results = {}
            for i, q in enumerate(self.quantiles):
                results[f"q{int(q*100):02d}"] = pred[:, i]
            
            res_df = pd.DataFrame(results)
            if 'timestamp' in df.columns:
                 res_df.index = df['timestamp'].iloc[-self.output_len:]
            else:
                 res_df.index = df.index[-self.output_len:]
            return res_df
