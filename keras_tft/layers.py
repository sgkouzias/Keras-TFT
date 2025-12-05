import keras
from keras import layers, ops, random


@keras.saving.register_keras_serializable()
class GatedLinearUnit(layers.Layer):
    """
    Gated Linear Unit (GLU).
    
    Implements the operation: GLU(x) = sigma(W1 x + b1) * (W2 x + b2).
    This allows the network to control the information flow.

    Attributes:
        hidden_dim (int): Dimension of the hidden layer.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1, **kwargs):
        """
        Initialize the GatedLinearUnit.

        Args:
            hidden_dim (int): Dimension of the hidden layer.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.linear = None
        self.gate = None
        self.dropout = None

    def build(self, input_shape):
        self.linear = layers.Dense(self.hidden_dim)
        self.gate = layers.Dense(self.hidden_dim, activation='sigmoid')
        self.dropout = layers.Dropout(self.dropout_rate)
        
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
    Can optionally accept a static context vector to condition the processing.
    
    Structure:
    x -> [Dense -> ELU -> Dense] -> GLU -> [Residual + Norm]
    
    If context is provided, it is projected and added to the first Dense layer's input.

    Attributes:
        hidden_dim (int): Dimension of the hidden layer.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1, **kwargs):
        """
        Initialize the GatedResidualNetwork.

        Args:
            hidden_dim (int): Dimension of the hidden layer.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.linear1 = None
        self.elu = layers.Activation('elu') # No weights, safe to keep
        self.linear2 = None
        self.dropout = None
        self.glu = None
        self.norm = None
        self.project = None
        self.context_proj = None

    def build(self, input_shape):
        self.linear1 = layers.Dense(self.hidden_dim)
        self.linear2 = layers.Dense(self.hidden_dim)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.glu = GatedLinearUnit(self.hidden_dim, self.dropout_rate)
        self.norm = layers.LayerNormalization()
        self.project = layers.Dense(self.hidden_dim)
        self.context_proj = layers.Dense(self.hidden_dim, use_bias=False)

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
            
            # x_in = x # Removed unused variable
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
    Variable Selection Network (VSN) for Static Features (2D inputs).
    
    Learns weights for each feature to suppress noise.
    
    Attributes:
        num_features (int): Number of input features.
        hidden_dim (int): Dimension of the hidden layer.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, num_features: int, hidden_dim: int, dropout_rate: float = 0.1, **kwargs):
        """
        Initialize the StaticVariableSelection layer.

        Args:
            num_features (int): Number of input features.
            hidden_dim (int): Dimension of the hidden layer.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            categorical_indices (List[int], optional): Indices of categorical features.
            vocab_sizes (List[int], optional): Vocabulary sizes for categorical features.
        """
        # Categorical handling
        self.categorical_indices = kwargs.pop('categorical_indices', [])
        self.vocab_sizes = kwargs.pop('vocab_sizes', [])
        
        super().__init__(**kwargs)
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.feature_projections = []
        self.feature_grns = []
        self.weight_grn = None
        self.softmax = None
        self.embedding_layers = []
    
    def build(self, input_shape):
        # input_shape: (Batch, Num_Features)
        
        self.feature_projections = [layers.Dense(self.hidden_dim) for _ in range(self.num_features)]
        self.feature_grns = [GatedResidualNetwork(self.hidden_dim, self.dropout_rate) 
                             for _ in range(self.num_features)]
        self.weight_grn = GatedResidualNetwork(self.num_features, self.dropout_rate)
        self.softmax = layers.Softmax(axis=-1)
        
        if self.categorical_indices:
            if not self.vocab_sizes or len(self.vocab_sizes) != len(self.categorical_indices):
                raise ValueError("vocab_sizes must be provided and match categorical_indices length")
            
            self.embedding_layers = []
            for vocab in self.vocab_sizes:
                self.embedding_layers.append(layers.Embedding(vocab, self.hidden_dim))

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
        if self.num_features > 1:
            weights = self.softmax(weights)  # (Batch, num_features)
        else:
            weights = ops.ones_like(weights)
        
        feature_list = ops.split(inputs, self.num_features, axis=-1)
        processed_features = []
        for i, feat in enumerate(feature_list):
            # Check if this feature index is categorical
            if i in self.categorical_indices:
                cat_idx = self.categorical_indices.index(i)
                embed = self.embedding_layers[cat_idx]
                
                # feat is (Batch, 1)
                # We need to cast to int for embedding
                feat_int = ops.cast(feat, "int32")

                # Validation: Clip to vocab size to prevent out-of-bounds
                if self.vocab_sizes: # Defensive check
                    vocab_limit = self.vocab_sizes[cat_idx]
                    feat_int = ops.clip(feat_int, 0, vocab_limit - 1)
                
                # Squeeze the last dim to get (Batch)
                feat_int = ops.squeeze(feat_int, axis=-1)
                
                proj = embed(feat_int) # (Batch, Hidden)
            else:
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
            "dropout_rate": self.dropout_rate,
            "categorical_indices": self.categorical_indices,
            "vocab_sizes": self.vocab_sizes
        })
        return config


@keras.saving.register_keras_serializable()
class MultivariateVariableSelection(layers.Layer):
    """
    Variable Selection Network (VSN) for Temporal Features (3D inputs).
    
    Learns weights for each feature to suppress noise.
    Can optionally accept a static context vector to condition the selection.
    
    Attributes:
        num_features (int): Number of input features.
        hidden_dim (int): Dimension of the hidden layer.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, num_features: int, hidden_dim: int, dropout_rate: float = 0.1, **kwargs):
        """
        Initialize the MultivariateVariableSelection layer.

        Args:
            num_features (int): Number of input features.
            hidden_dim (int): Dimension of the hidden layer.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            categorical_indices (List[int], optional): Indices of categorical features.
            vocab_sizes (List[int], optional): Vocabulary sizes for categorical features.
        """
        # Categorical handling
        self.categorical_indices = kwargs.pop('categorical_indices', [])
        self.vocab_sizes = kwargs.pop('vocab_sizes', [])
        
        super().__init__(**kwargs)
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.feature_grns = []
        self.weight_grn = None
        self.softmax = None
        self.input_projections = []
        self.embedding_layers = []

    def build(self, input_shape):
        self.feature_grns = [
            GatedResidualNetwork(self.hidden_dim, self.dropout_rate) 
            for _ in range(self.num_features)
        ]
        self.weight_grn = GatedResidualNetwork(self.num_features, self.dropout_rate)
        self.softmax = layers.Softmax(axis=-1)
        
        self.input_projections = [
            layers.Dense(self.hidden_dim) for _ in range(self.num_features)
        ]
        
        if self.categorical_indices:
            if not self.vocab_sizes or len(self.vocab_sizes) != len(self.categorical_indices):
                raise ValueError("vocab_sizes must be provided and match categorical_indices length")
            
            self.embedding_layers = []
            for vocab in self.vocab_sizes:
                self.embedding_layers.append(layers.Embedding(vocab, self.hidden_dim))

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
            
        if self.num_features > 1:
            weights = self.softmax(weights) # (Batch, Num_Features)
        else:
            # If 1 feature, weight is 1.0
            weights = ops.ones_like(weights)
        
        # 2. Process features
        feature_list = ops.split(x, self.num_features, axis=-1)
        processed_features = []
        
        for i, feat in enumerate(feature_list):
            # Check if this feature index is categorical
            if i in self.categorical_indices:
                cat_idx = self.categorical_indices.index(i)
                embed = self.embedding_layers[cat_idx]
                
                # feat is (Batch, Time, 1) or (Batch, 1)
                # We need to cast to int for embedding
                feat_int = ops.cast(feat, "int32")
                
                # Validation: Check for invalid indices
                # Note: This adds ops to the graph
                # tf.debugging.assert_non_negative(feat_int, message=f"Negative index in categorical feature {i}")
                # We assume correct preprocessing, but this check catches raw float errors
                
                # Validation: Clip to vocab size to prevent out-of-bounds
                if self.vocab_sizes:
                    vocab_limit = self.vocab_sizes[cat_idx]
                    feat_int = ops.clip(feat_int, 0, vocab_limit - 1)
                
                # Squeeze the last dim to get (Batch, Time)
                feat_int = ops.squeeze(feat_int, axis=-1)
                
                feat_proj = embed(feat_int) # (Batch, Time, Hidden)
            else:
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
            "dropout_rate": self.dropout_rate,
            "categorical_indices": self.categorical_indices,
            "vocab_sizes": self.vocab_sizes
        })
        return config


@keras.saving.register_keras_serializable()
class InterpretableMultiHeadAttention(layers.Layer):
    """
    Interpretable Multi-Head Attention (IMHA) from TFT paper.
    Interpretable Multi-Head Attention Mechanism.

    This layer implements Multi-Head Attention with a specific design that allows for proper
    interpretation of attention weights, maintaining the output dimension equal to the model dimension
    to facilitate residual connections.

    Attributes:
        num_heads (int): Number of attention heads.
        key_dim (int): Dimensionality of the key/query projections per head.
        output_dim (int): Dimensionality of the final output projection (usually hidden_dim).
        dropout (float): Dropout rate.
    """
    def __init__(self, num_heads: int, key_dim: int, output_dim: int, dropout: float = 0.0, **kwargs):
        """
        Initialize the InterpretableMultiHeadAttention layer.

        Args:
            num_heads (int): Number of attention heads.
            key_dim (int): Size of each attention head for query/key.
            output_dim (int): Size of the final output (and value input).
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.output_dim = output_dim # Explicitly set output_dim
        self.dropout = dropout
        
        self.query_dense = layers.Dense(key_dim * num_heads)
        self.key_dense = layers.Dense(key_dim * num_heads)
        self.value_dense = layers.Dense(key_dim) # Shared value projection
        
        self.output_dense = None # Initialize in build to ensure correct output dim
        self.dropout_layer = layers.Dropout(dropout)
        
        # We need to ensure output_dense projects back to the EXPECTED hidden dimension (e.g. 128)
        # In the previous implementation, we projected to num_heads * key_dim.
        # If hidden_dim=100, num_heads=3 -> key_dim=33 -> output=99 != 100.
        # We need to know the target hidden_dim.
        # We can accept it as an argument or infer it? 
        # Standard Keras MHA accepts 'key_dim' but also 'value_dim' or assumes output matches input or specific arg.
        # Here we follow the user's fix: project to self.hidden_dim.
        # But we didn't save self.hidden_dim in __init__ properly (we set self.hidden_dim = key_dim).
        # We should probably accept hidden_dim in __init__ to be safe, or assume the user wants 
        # the projection to match (num_heads * key_dim) IF consistent,
        # BUT the user explicitly requested: "Change the output projection to explicitly target hidden_dim"
        # and checking the init sig, we don't pass hidden_dim.
        # However, `model.py` passes `key_dim = hidden_dim // num_heads`.
        # So we can try to recover hidden_dim if we assume the user wants to restore it?
        # A safer bet is to accept `hidden_dim` in `__init__` if possible.
        # Looking at `model.py` call:
        # attn_layer = InterpretableMultiHeadAttention(
        #     num_heads=self.num_heads, key_dim=self.hidden_dim // self.num_heads, dropout=self.dropout_rate
        # )
        # It does NOT pass hidden_dim.
        # So we can't magically know '100' if we receive '33'.
        # However, we can modify `model.py` to pass it.
        # AND/OR we can modify `__init__` here.
        # Let's modify `__init__` to accept `output_dim` or `hidden_dim`?
        # OR we can assume `output_dense` size should simply be independent of key_dim*num_heads.
        # But we need to know the size.
        # I will modify `model.py` to pass `hidden_dim` to `InterpretableMultiHeadAttention`.
        # And update `__init__` here to accept it.
        # self.output_dim = kwargs.get('output_dim', key_dim * num_heads) # Moved to top


        
    def build(self, input_shape):
        # input_shape can be a list [q_shape, k_shape, v_shape] or single shape
        if isinstance(input_shape, list):
            q_shape = input_shape[0] # (B, T_q, D)
            k_shape = input_shape[1]
            v_shape = input_shape[2]
        else:
            q_shape = k_shape = v_shape = input_shape
            
        # Build sublayers explicitly
        # Inputs to q, k dense are (B, T, D)
        self.query_dense.build(q_shape)
        self.key_dense.build(k_shape)
        self.value_dense.build(v_shape)
        
        # Output dense takes concatenated heads: (B, T, Num_Heads * Key_Dim)
        # We want to project back to the hidden dimension (output_dim).
        concat_last_dim = self.num_heads * self.key_dim
        self.output_dense = layers.Dense(self.output_dim) 
        
        # We need to construct the input shape for output_dense
        # It takes (B, T_q, H*D)
        # Use None for Batch and Time dimensions for safety
        output_dense_input_shape = (None, None, concat_last_dim)
        self.output_dense.build(output_dense_input_shape)
        
        super().build(input_shape)
        
    def call(self, query, key, value, return_attention_scores=False):
        # query: (B, T_q, D)
        # key:   (B, T_k, D)
        # value: (B, T_k, D)
        
        B = ops.shape(query)[0]
        T_q = ops.shape(query)[1]
        T_k = ops.shape(key)[1]
        
        # Project Q, K
        # (B, T, H * D_k)
        q = self.query_dense(query)
        k = self.key_dense(key)
        
        # Project V (Shared)
        # (B, T, D_v) -> We use key_dim as D_v
        v = self.value_dense(value)
        
        # Reshape Q, K for heads
        # (B, T, H, D_k)
        q = ops.reshape(q, (B, T_q, self.num_heads, self.key_dim))
        k = ops.reshape(k, (B, T_k, self.num_heads, self.key_dim))
        
        # Transpose for attention: (B, H, T, D)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        
        # Scaled Dot-Product Attention
        # (B, H, T_q, D) @ (B, H, D, T_k) -> (B, H, T_q, T_k)
        score = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        scale = ops.cast(self.key_dim, score.dtype) ** -0.5
        score = score * scale
        
        # Masking could be added here if needed (e.g. causal), but TFT usually masks future in data prep
        # or uses decoder masking. For now assuming no mask or passed externally?
        # Standard MHA supports mask. We should support it too if passed.
        # But for now, simple implementation.
        
        attn_weights = layers.Softmax(axis=-1)(score)
        
        # Apply attention to V
        # V is (B, T_k, D_v). We need to broadcast or repeat for heads?
        # Paper: "We modify MHA to share values in each head."
        # So Head_h = Attention(Q_h, K_h, V_shared)
        # Result_h = attn_weights_h @ V_shared
        # attn_weights: (B, H, T_q, T_k)
        # V: (B, T_k, D_v)
        
        # We can treat V as having 1 head: (B, 1, T_k, D_v)
        v_reshaped = ops.expand_dims(v, axis=1)
        
        # (B, H, T_q, T_k) @ (B, 1, T_k, D_v) -> (B, H, T_q, D_v)
        # Broadcasting works on axis 1?
        # Matmul on last 2 dims.
        # (T_q, T_k) @ (T_k, D_v) -> (T_q, D_v)
        # Batch dims: (B, H) vs (B, 1) -> (B, H)
        output = ops.matmul(attn_weights, v_reshaped)
        
        # Concatenate heads? 
        # Paper: "A linear layer combines the outputs of all heads: H = Linear(Concat(Head_1, ... Head_H))"
        # output is (B, H, T_q, D_v)
        # Reshape to (B, T_q, H * D_v)
        output = ops.transpose(output, (0, 2, 1, 3))
        # Concatenate heads: (B, T_q, H, D_v) -> (B, T_q, H * D_v)
        output = ops.reshape(output, (B, T_q, self.num_heads * self.key_dim))
        
        # Project back to key_dim (which is hidden_dim)
        output = self.output_dense(output)
        output = self.dropout_layer(output)
        
        if return_attention_scores:
            return output, attn_weights
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout": self.dropout,
            "output_dim": self.output_dim
        })
        return config

