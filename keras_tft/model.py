import keras
from keras import layers
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from .layers import GatedResidualNetwork, MultivariateVariableSelection, GatedLinearUnit, StaticVariableSelection
from .loss import QuantileLoss

class TFTForecaster:
    """
    TFT Wrapper supporting Future Covariates and Encoder-Decoder structure.
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
        learning_rate: float = 0.001
    ):
        self.input_len = input_chunk_length
        self.output_len = output_chunk_length
        self.quantiles = quantiles
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        
        self.model = None
        self.scalers: Dict[str, Tuple[float, float]] = {}
        self.target_col: Optional[str] = None
        self.past_cov_cols: List[str] = []
        self.future_cov_cols: List[str] = []
        self.static_cov_cols: List[str] = []
        self.feature_cols: List[str] = []

    def _build_model(self, num_past_features: int, num_future_features: int, num_static_features: int) -> keras.Model:
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
                num_static_features, self.hidden_dim, self.dropout_rate, name="vsn_static"
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
        x_past, past_weights = MultivariateVariableSelection(
            num_past_features, self.hidden_dim, self.dropout_rate, name="vsn_past",
        )([input_past, c_s])
        
        # Future VSN
        # Use c_s for context
        x_fut, future_weights = MultivariateVariableSelection(
            num_future_features, self.hidden_dim, self.dropout_rate, name="vsn_future"
        )([input_future, c_s])

        # 2. LSTM Encoder-Decoder (Seq2Seq)
        lstm_layer = layers.LSTM(self.hidden_dim, return_sequences=True, return_state=True)
        
        # Initialize LSTM state with c_h, c_c
        initial_state = [c_h, c_c]

        # Static Enrichment for Past (use c_e)
        x_past = GatedResidualNetwork(self.hidden_dim, self.dropout_rate, name="grn_enrich_past")([x_past, c_e])
        
        # Run LSTM on Past
        encoder_out, state_h, state_c = lstm_layer(x_past, initial_state=initial_state)
        
        # Post-LSTM Gate (GLU) + Add + Norm
        encoder_out = GatedLinearUnit(self.hidden_dim, self.dropout_rate)(encoder_out)
        encoder_out = layers.LayerNormalization()(encoder_out + x_past) # Residual from x_past

        # Static Enrichment for Future (use c_e)
        x_fut = GatedResidualNetwork(self.hidden_dim, self.dropout_rate, name="grn_enrich_fut")([x_fut, c_e])

        # Run LSTM on Future
        # We initialize with encoder state
        decoder_out, _, _ = lstm_layer(x_fut, initial_state=[state_h, state_c])
        
        # Post-LSTM Gate + Add + Norm
        decoder_out = GatedLinearUnit(self.hidden_dim, self.dropout_rate)(decoder_out)
        decoder_out = layers.LayerNormalization()(decoder_out + x_fut)

        # 3. Multi-Head Attention
        attn_out = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.hidden_dim
        )(query=decoder_out, value=encoder_out, key=encoder_out)
        
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
        
        model = keras.Model(inputs=inputs, outputs=predictions)
        
        # Create Explainability Model (sharing layers)
        # We expose weights
        explain_outputs = {
            'predictions': predictions,
            'past_weights': past_weights,
            'future_weights': future_weights
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

        model.compile(optimizer=opt, loss=QuantileLoss(self.quantiles))
        return model

    def get_feature_importance(self, df: pd.DataFrame):
        """Extract feature importance using explainability model."""
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

    def _scale_matrix(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scales data and returns a DataFrame for easier column slicing."""
        data_dict = {}
        for col in self.feature_cols:
            val = df[col].values.astype(float)
            if fit:
                mean, std = np.mean(val), np.std(val)
                if std == 0: std = 1e-7
                self.scalers[col] = (mean, std)
            
            if col not in self.scalers:
                raise ValueError(f"Column {col} not found in fitted scalers.")
                
            mean, std = self.scalers[col]
            data_dict[col] = (val - mean) / (std + 1e-7)
            
        return pd.DataFrame(data_dict, index=df.index)

    def fit(self, df, target_col, past_cov_cols=None, future_cov_cols=None, static_cov_cols=None,
            epochs=10, batch_size=32, verbose=1,
            validation_split=0.0,
            use_lr_schedule=True, 
            use_early_stopping=False, early_stopping_patience=10):
        
        self.target_col = target_col
        self.past_cov_cols = past_cov_cols if past_cov_cols else []
        self.future_cov_cols = future_cov_cols if future_cov_cols else []
        self.static_cov_cols = static_cov_cols if static_cov_cols else []
        
        # All columns needed
        self.feature_cols = list(set([target_col] + self.past_cov_cols + self.future_cov_cols + self.static_cov_cols))
        
        # Scale
        matrix_df = self._scale_matrix(df, fit=True)
        
        # Prepare Feature Lists
        # Past Input: Target + Past Covs + Future Covs (as observed in past)
        past_cols = [target_col] + self.past_cov_cols + self.future_cov_cols
        # Future Input: Future Covs only
        fut_cols = self.future_cov_cols
        # Static Input
        static_cols = self.static_cov_cols
        
        # Create Windows
        X_past_list, X_fut_list, X_static_list, y_list = [], [], [], []
        
        matrix_vals = matrix_df.values
        col_to_idx = {name: i for i, name in enumerate(matrix_df.columns)}
        
        past_idxs = [col_to_idx[c] for c in past_cols]
        fut_idxs = [col_to_idx[c] for c in fut_cols]
        static_idxs = [col_to_idx[c] for c in static_cols]
        target_idx = col_to_idx[target_col]
        
        num_samples = len(matrix_df) - self.input_len - self.output_len
        if num_samples <= 0: raise ValueError("Dataframe too short.")

        for i in range(num_samples):
            # Past: [t : t+input]
            X_past_list.append(matrix_vals[i : i+self.input_len, past_idxs])
            # Future: [t+input : t+input+output]
            X_fut_list.append(matrix_vals[i+self.input_len : i+self.input_len+self.output_len, fut_idxs])
            # Static: [i] (Assuming static features are constant or we take the value at start of window)
            # If truly static, any row works. If they change slowly, taking start is fine.
            if static_idxs:
                X_static_list.append(matrix_vals[i, static_idxs])
            else:
                X_static_list.append(np.zeros(0)) # Empty
                
            # Target: [t+input : t+input+output]
            y_list.append(matrix_vals[i+self.input_len : i+self.input_len+self.output_len, target_idx])
            
        X_past = np.array(X_past_list)
        X_fut = np.array(X_fut_list)
        X_static = np.array(X_static_list)
        y = np.array(y_list)[..., np.newaxis] # Expand for quantiles
        
        if self.model is None:
            self.model = self._build_model(
                num_past_features=len(past_cols),
                num_future_features=len(fut_cols),
                num_static_features=len(static_cols)
            )
            
        if verbose > 0:
            print(f"Training on {len(X_past)} samples. Past-covariates: {len(past_cols)}, Future-covariates: {len(fut_cols)}, Static-covariates: {len(static_cols)}")
        
        inputs = [X_past, X_fut]
        if static_cols:
            inputs.append(X_static)
        
        # Callbacks
        callbacks = []
        if use_lr_schedule:
            lr_schedule = keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=verbose
            )
            callbacks.append(lr_schedule)
        
        if use_early_stopping:
            monitor = 'val_loss' if validation_split > 0 else 'loss'
            early_stop = keras.callbacks.EarlyStopping(
                monitor=monitor, patience=early_stopping_patience, restore_best_weights=True, verbose=verbose
            )
            callbacks.append(early_stop)
            
        self.model.fit(inputs, y, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                       validation_split=validation_split, callbacks=callbacks)

    def summary(self):
        """Prints the summary of the underlying Keras model."""
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet. Call fit() first.")

    def predict(self, df):
        """
        Expects a dataframe containing history AND the future horizon rows for future covariates.
        """
        if self.model is None: raise ValueError("Model not fitted.")
        
        # Scale
        matrix_df = self._scale_matrix(df, fit=False)
        matrix_vals = matrix_df.values
        col_to_idx = {name: i for i, name in enumerate(matrix_df.columns)}
        
        past_cols = [self.target_col] + self.past_cov_cols + self.future_cov_cols
        fut_cols = self.future_cov_cols
        static_cols = self.static_cov_cols
        
        past_idxs = [col_to_idx[c] for c in past_cols]
        fut_idxs = [col_to_idx[c] for c in fut_cols]
        static_idxs = [col_to_idx[c] for c in static_cols]
        
        # We need the LAST valid sequence set
        # Slice Past: [-input_len - output_len : -output_len]? 
        # No, usually predict is called with data UP TO start of prediction + Future rows
        
        # Assumption: df ends exactly at 'current_time + output_len'
        # So past is [-input_len-output_len : -output_len]
        # And future is [-output_len:]
        
        total_required = self.input_len + self.output_len
        if len(df) < total_required:
            raise ValueError(f"DF length {len(df)} < required {total_required}")
            
        # Extract single batch
        # 1. Past: The data BEFORE the forecast horizon
        past_seq = matrix_vals[-(self.input_len + self.output_len) : -self.output_len, past_idxs]
        
        # 2. Future: The data DURING the forecast horizon
        fut_seq = matrix_vals[-self.output_len:, fut_idxs]
        
        # 3. Static: Take from the last row (or any row)
        if static_idxs:
            static_seq = matrix_vals[-1, static_idxs]
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
        
        results = {}
        for i, q in enumerate(self.quantiles):
            results[f"q{int(q*100)}"] = pred_actual[0, :, i]
        return pd.DataFrame(results)
