import keras
from keras import ops
from typing import List

@keras.saving.register_keras_serializable()
class QuantileLoss(keras.losses.Loss):
    """
    Calculates Pinball Loss (Quantile Loss) for probabilistic forecasting.
    
    The loss is calculated as:
    L(y, y_pred) = max(q * (y - y_pred), (q - 1) * (y - y_pred))
    averaged over all quantiles and samples.

    Attributes:
        quantiles (List[float]): List of quantiles to optimize for.
    """
    def __init__(self, quantiles: List[float], **kwargs):
        """
        Initialize the QuantileLoss.

        Args:
            quantiles (List[float]): List of quantiles (e.g., [0.1, 0.5, 0.9]).
        """
        super().__init__(**kwargs)
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        
        # y_pred: (Batch, Output_len, Quantiles)
        # y_true: (Batch, Output_len, 1) or (Batch, Output_len)
        if y_pred.shape[-1] != len(self.quantiles):
             raise ValueError(f"Expected {len(self.quantiles)} quantiles, got {y_pred.shape[-1]}")
        
        if ops.ndim(y_true) == 2:
            y_true = ops.expand_dims(y_true, axis=-1)
            
        # Broadcast y_true to match num_quantiles
        # (Batch, Time, 1) -> (Batch, Time, Quantiles)
        y_true = ops.broadcast_to(y_true, ops.shape(y_pred))
        
        loss_list = []
        
        # Calculate loss for each quantile
        error = y_true - y_pred
        
        for i, q in enumerate(self.quantiles):
            q_error = error[..., i]
            loss_q = ops.maximum(q * q_error, (q - 1) * q_error)
            loss_list.append(loss_q)
            
        return ops.mean(ops.stack(loss_list, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"quantiles": self.quantiles})
        return config
