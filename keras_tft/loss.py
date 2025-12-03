import keras
from keras import ops
from typing import List

@keras.saving.register_keras_serializable()
class QuantileLoss(keras.losses.Loss):
    """
    Calculates Pinball Loss for probabilistic forecasting.
    """
    def __init__(self, quantiles: List[float], **kwargs):
        super().__init__(**kwargs)
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        
        if len(y_true.shape) == 2:
            y_true = ops.expand_dims(y_true, axis=-1)
            
        error = y_true - y_pred
        loss_list = []
        
        for i, q in enumerate(self.quantiles):
            q_error = error[..., i]
            loss_q = ops.maximum(q * q_error, (q - 1) * q_error)
            loss_list.append(loss_q)
            
        return ops.mean(ops.stack(loss_list, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"quantiles": self.quantiles})
        return config
