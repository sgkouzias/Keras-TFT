import keras
from keras import layers, ops

@keras.saving.register_keras_serializable()
class GatedLinearUnit(layers.Layer):
    """
    Gated Linear Unit (GLU).
    GLU(x) = sigma(W1 x + b1) * (W2 x + b2)
    """
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.linear = layers.Dense(hidden_dim)
        self.gate = layers.Dense(hidden_dim, activation='sigmoid')
        self.dropout = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        self.linear.build(input_shape)
        self.gate.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        x = self.linear(inputs)
        g = self.gate(inputs)
        return self.dropout(x * g)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


@keras.saving.register_keras_serializable()
class GatedResidualNetwork(layers.Layer):
    """
    Gated Residual Network (GRN).
    Applies non-linear processing with gating and residual connection.
    Can optionally accept a static context vector.
    """
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.linear1 = layers.Dense(hidden_dim)
        self.elu = layers.Activation('elu')
        self.linear2 = layers.Dense(hidden_dim)
        self.dropout = layers.Dropout(dropout_rate)
        self.glu = GatedLinearUnit(hidden_dim, dropout_rate)
        self.norm = layers.LayerNormalization()
        self.project = layers.Dense(hidden_dim)
        
        # Context projection
        self.context_proj = layers.Dense(hidden_dim, use_bias=False)

    def build(self, input_shape):
        # input_shape can be a list if context is provided: [input_shape, context_shape]
        if isinstance(input_shape, list):
            x_shape = input_shape[0]
            c_shape = input_shape[1]
            self.context_proj.build(c_shape)
        else:
            x_shape = input_shape
            
        self.linear1.build(x_shape)
        self.linear2.build(x_shape)
        self.glu.build(x_shape)
        
        if x_shape[-1] != self.hidden_dim:
            self.project.build(x_shape)
            
        self.norm.build(x_shape)
        super().build(input_shape)

    def call(self, inputs):
        if isinstance(inputs, list):
            x, context = inputs
            # Project context and add to linear1 input
            c = self.context_proj(context)
            # If x has time dim and c doesn't, expand c
            if len(x.shape) == 3 and len(c.shape) == 2:
                c = ops.expand_dims(c, axis=1)
            
            x_in = x
            residual = x
            
            # Feed Forward with Context
            x = self.linear1(x)
            x = x + c # Add context
            x = self.elu(x)
            x = self.linear2(x)
            x = self.dropout(x)
            
            # Gating
            x = self.glu(x)
            
        else:
            x = inputs
            residual = x
            
            # Feed Forward
            x = self.linear1(x)
            x = self.elu(x)
            x = self.linear2(x)
            x = self.dropout(x)
            
            # Gating
            x = self.glu(x)
        
        # Residual Connection
        if residual.shape[-1] != self.hidden_dim:
            residual = self.project(residual)
            
        return self.norm(residual + x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim, 
            "dropout_rate": self.dropout_rate
        })
        return config


@keras.saving.register_keras_serializable()
class StaticVariableSelection(layers.Layer):
    """
    Variable Selection Network for Static Features (2D inputs).
    """
    def __init__(self, num_features: int, hidden_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.feature_projections = [layers.Dense(hidden_dim) for _ in range(num_features)]
        self.feature_grns = [GatedResidualNetwork(hidden_dim, dropout_rate) 
                             for _ in range(num_features)]
        self.weight_grn = GatedResidualNetwork(num_features, dropout_rate)
        self.softmax = layers.Softmax(axis=-1)
    
    def build(self, input_shape):
        # input_shape: (Batch, Num_Features)
        self.weight_grn.build(input_shape)
        
        single_feature_shape = list(input_shape)
        single_feature_shape[-1] = 1
        single_feature_shape = tuple(single_feature_shape)
        
        processed_shape = list(input_shape)
        processed_shape[-1] = self.hidden_dim
        processed_shape = tuple(processed_shape)

        for i in range(self.num_features):
            self.feature_projections[i].build(single_feature_shape)
            self.feature_grns[i].build(processed_shape)
            
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (Batch, Features)
        weights = self.weight_grn(inputs)
        weights = self.softmax(weights)  # (Batch, num_features)
        
        feature_list = ops.split(inputs, self.num_features, axis=-1)
        processed_features = []
        for i, feat in enumerate(feature_list):
            proj = self.feature_projections[i](feat)
            processed = self.feature_grns[i](proj)
            processed_features.append(processed)
        
        processed_stack = ops.stack(processed_features, axis=1)  # (Batch, num_features, hidden_dim)
        weights_exp = ops.expand_dims(weights, axis=-1)  # (Batch, num_features, 1)
        
        weighted_sum = ops.sum(processed_stack * weights_exp, axis=1)  # (Batch, hidden_dim)
        return weighted_sum, weights # Return weights for interpretability

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


@keras.saving.register_keras_serializable()
class MultivariateVariableSelection(layers.Layer):
    """
    Variable Selection Network (VSN) for Temporal Features (3D inputs).
    Learns weights for each feature to suppress noise.
    Accepts optional static context.
    """
    def __init__(self, num_features: int, hidden_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.feature_grns = [
            GatedResidualNetwork(hidden_dim, dropout_rate) 
            for _ in range(num_features)
        ]
        self.weight_grn = GatedResidualNetwork(num_features, dropout_rate)
        self.softmax = layers.Softmax(axis=-1)
        
        self.input_projections = [
            layers.Dense(hidden_dim) for _ in range(num_features)
        ]
        # Removed flatten_context Dense layer, using ops.mean instead

    def build(self, input_shape):
        # input_shape: (Batch, Time, Num_Features) or [(Batch, Time, Num_Features), (Batch, Context_Dim)]
        if isinstance(input_shape, list):
            x_shape = input_shape[0]
            c_shape = input_shape[1]
            # Weight GRN input: flattened x (Batch, Num_Features) + context (Batch, Context)
            # Actually weight_grn takes a list [x_flat, context]
            # x_flat shape is (Batch, Num_Features)
            self.weight_grn.build([(x_shape[0], self.num_features), c_shape])
        else:
            x_shape = input_shape
            self.weight_grn.build((x_shape[0], self.num_features))
            
        # Shape logic for per-feature processing
        single_feature_shape = list(x_shape)
        single_feature_shape[-1] = 1
        single_feature_shape = tuple(single_feature_shape)
        
        processed_shape = list(x_shape)
        processed_shape[-1] = self.hidden_dim
        processed_shape = tuple(processed_shape)

        for i in range(self.num_features):
            self.input_projections[i].build(single_feature_shape)
            self.feature_grns[i].build(processed_shape)
            
        super().build(input_shape)

    def call(self, inputs):
        context = None
        if isinstance(inputs, list):
            x, context = inputs
        else:
            x = inputs
            
        # 1. Weights
        # Average across time: (Batch, Time, Features) -> (Batch, Features)
        flattened = ops.mean(x, axis=1)
        
        if context is not None:
            weights = self.weight_grn([flattened, context])
        else:
            weights = self.weight_grn(flattened)
            
        weights = self.softmax(weights) # (Batch, Num_Features)
        
        # 2. Process features
        feature_list = ops.split(x, self.num_features, axis=-1)
        processed_features = []
        
        for i, feat in enumerate(feature_list):
            feat_proj = self.input_projections[i](feat)
            processed = self.feature_grns[i](feat_proj)
            processed_features.append(processed)
            
        processed_stack = ops.stack(processed_features, axis=-2) # (Batch, Time, Num_Features, Hidden)
        
        # 3. Weighted Sum
        # weights: (Batch, Num_Features)
        # We need to expand weights to (Batch, 1, Num_Features, 1) for broadcasting?
        # processed_stack: (Batch, Time, Num_Features, Hidden)
        # We want to sum over Num_Features axis (-2).
        
        weights_expanded = ops.expand_dims(weights, axis=1) # (Batch, 1, Num_Features)
        weights_expanded = ops.expand_dims(weights_expanded, axis=-1) # (Batch, 1, Num_Features, 1)
        
        weighted_sum = ops.sum(processed_stack * weights_expanded, axis=-2) # (Batch, Time, Hidden)
        
        return weighted_sum, weights # Return weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate
        })
        return config
