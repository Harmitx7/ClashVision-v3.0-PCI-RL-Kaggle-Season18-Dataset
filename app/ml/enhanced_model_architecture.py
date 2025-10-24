import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EnhancedTransformerLSTMModel(tf.keras.Model):
    """
    Enhanced Hybrid Transformer-LSTM model with PCI integration and RL loop
    ClashVision v3.0-PCI-RL
    """
    
    def __init__(
        self,
        sequence_length: int = 50,
        feature_dim: int = 64,
        d_model: int = 128,
        num_heads: int = 8,
        num_transformer_layers: int = 4,
        lstm_units: int = 64,
        dropout_rate: float = 0.1,
        pci_embedding_dim: int = 16,
        enable_rl_loop: bool = True,
        **kwargs
    ):
        super(EnhancedTransformerLSTMModel, self).__init__(**kwargs)
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.pci_embedding_dim = pci_embedding_dim
        self.enable_rl_loop = enable_rl_loop
        
        # Input projections
        self.feature_projection = layers.Dense(d_model, activation='relu', name='feature_projection')
        self.pci_embedding = layers.Dense(pci_embedding_dim, activation='relu', name='pci_embedding')
        self.input_dropout = layers.Dropout(dropout_rate)
        
        # PCI conditioning layer
        self.pci_conditioning = PCIConditioningLayer(d_model, pci_embedding_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(sequence_length, d_model)
        
        # Enhanced Transformer layers with PCI awareness
        self.transformer_layers = []
        for i in range(num_transformer_layers):
            self.transformer_layers.append(
                PCIAwareTransformerBlock(
                    d_model, num_heads, dropout_rate, name=f'transformer_block_{i}'
                )
            )
        
        # LSTM layers with temporal momentum learning
        self.lstm1 = layers.LSTM(
            lstm_units, 
            return_sequences=True, 
            dropout=dropout_rate,
            name='lstm_temporal_1'
        )
        self.lstm2 = layers.LSTM(
            lstm_units // 2, 
            return_sequences=False, 
            dropout=dropout_rate,
            name='lstm_temporal_2'
        )
        
        # Ensemble fusion layer
        self.ensemble_fusion = EnsembleFusionLayer(d_model, lstm_units // 2)
        
        # Output layers with confidence estimation
        self.dense1 = layers.Dense(64, activation='relu', name='dense_1')
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(32, activation='relu', name='dense_2')
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Dual output: prediction + confidence
        self.prediction_head = layers.Dense(1, activation='sigmoid', name='prediction')
        self.confidence_head = layers.Dense(1, activation='sigmoid', name='confidence')
        
        # Batch normalization
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()
        
        # Reinforcement learning components
        if enable_rl_loop:
            self.rl_weight_adjustment = RLWeightAdjustmentLayer()
    
    def build(self, input_shape):
        """Build the model layers"""
        if isinstance(input_shape, list):
            feature_shape, pci_shape = input_shape
        else:
            feature_shape = input_shape
            pci_shape = (None, 1)
        
        # Build all layers
        super().build(input_shape)
        return
    
    def call(self, inputs, training=None, mask=None):
        """Enhanced forward pass with PCI conditioning"""
        # Unpack inputs: [features, pci_value]
        if isinstance(inputs, list) and len(inputs) == 2:
            feature_input, pci_input = inputs
        else:
            feature_input = inputs
            pci_input = tf.ones((tf.shape(inputs)[0], 1)) * 0.5  # Default PCI
        
        # Project features to model dimension
        x = self.feature_projection(feature_input)
        x = self.input_dropout(x, training=training)
        
        # Embed PCI
        pci_embedded = self.pci_embedding(pci_input)
        
        # Apply PCI conditioning
        x = self.pci_conditioning([x, pci_embedded])
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Store transformer output for ensemble
        transformer_output = x
        
        # Apply PCI-aware transformer layers
        for transformer_layer in self.transformer_layers:
            transformer_output = transformer_layer([transformer_output, pci_embedded], training=training)
        
        # Apply LSTM layers for temporal learning
        lstm_output = self.lstm1(x, training=training)
        lstm_output = self.lstm2(lstm_output, training=training)
        
        # Ensemble fusion with Bayesian weighting
        fused_output = self.ensemble_fusion([transformer_output, lstm_output, pci_embedded])
        
        # Apply dense layers
        x = self.dense1(fused_output)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)
        
        # Dual output heads
        prediction = self.prediction_head(x)
        confidence = self.confidence_head(x)
        
        # Apply RL weight adjustment if enabled
        if self.enable_rl_loop and hasattr(self, 'rl_weight_adjustment'):
            prediction = self.rl_weight_adjustment([prediction, pci_input])
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'pci_conditioned': True
        }
    
    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_transformer_layers': self.num_transformer_layers,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'pci_embedding_dim': self.pci_embedding_dim,
            'enable_rl_loop': self.enable_rl_loop
        })
        return config

class PCIConditioningLayer(layers.Layer):
    """Layer to condition transformer input with PCI embedding"""
    
    def __init__(self, d_model: int, pci_embedding_dim: int, **kwargs):
        super(PCIConditioningLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.pci_embedding_dim = pci_embedding_dim
        
        # Conditioning transformation
        self.conditioning_transform = layers.Dense(d_model, activation='tanh')
        
    def call(self, inputs):
        """Apply PCI conditioning to features"""
        features, pci_embedded = inputs
        
        # Expand PCI embedding to match sequence length
        seq_len = tf.shape(features)[1]
        pci_expanded = tf.expand_dims(pci_embedded, axis=1)
        pci_expanded = tf.tile(pci_expanded, [1, seq_len, 1])
        
        # Transform PCI for conditioning
        pci_conditioning = self.conditioning_transform(pci_expanded)
        
        # Apply multiplicative conditioning
        conditioned_features = features * (1.0 + pci_conditioning)
        
        return conditioned_features
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'pci_embedding_dim': self.pci_embedding_dim
        })
        return config

class PCIAwareTransformerBlock(layers.Layer):
    """Enhanced transformer block with PCI awareness"""
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1, **kwargs):
        super(PCIAwareTransformerBlock, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Multi-head attention with PCI modulation
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        # PCI-modulated attention weights
        self.pci_attention_modulation = layers.Dense(num_heads, activation='softmax')
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_model * 4, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model)
        ])
        
        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=None, mask=None):
        """Forward pass with PCI-aware attention"""
        features, pci_embedded = inputs
        
        # Generate PCI-modulated attention weights
        attention_weights = self.pci_attention_modulation(pci_embedded)
        
        # Multi-head attention - fix the mask issue
        attn_output = self.multi_head_attention(
            query=features, 
            value=features, 
            key=features,
            attention_mask=None,  # Set to None instead of mask to avoid tensor conversion issues
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(features + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config

class EnsembleFusionLayer(layers.Layer):
    """Weighted Bayesian fusion of Transformer and LSTM outputs with PCI conditioning"""
    
    def __init__(self, transformer_dim=128, lstm_dim=32, output_dim=64, **kwargs):
        super(EnsembleFusionLayer, self).__init__(**kwargs)
        self.transformer_dim = transformer_dim
        self.lstm_dim = lstm_dim
        self.output_dim = output_dim
        
        # Initialize layers in __init__ but don't build them yet
        self.pci_embedding = None
        self.pci_weight_generator = None
        self.lstm_projection = None
        self.output_projection = None
    
    def build(self, input_shape):
        """Build the layer - create all sublayers here"""
        super(EnsembleFusionLayer, self).build(input_shape)
        
        # PCI embedding for adaptive fusion
        self.pci_embedding = tf.keras.layers.Dense(16, activation='relu', name='pci_embedding')
        self.pci_weight_generator = tf.keras.layers.Dense(2, activation='softmax', name='pci_weights')
        
        # Projection layers
        self.lstm_projection = tf.keras.layers.Dense(self.transformer_dim, name='lstm_projection')
        self.output_projection = tf.keras.layers.Dense(self.output_dim, activation='relu', name='output_projection')
        
        # Build the sublayers
        self.pci_embedding.build((None, 1))
        self.pci_weight_generator.build((None, 16))
        self.lstm_projection.build((None, self.lstm_dim))
        self.output_projection.build((None, self.transformer_dim))
        
    def call(self, inputs):
        """Fuse transformer and LSTM outputs with PCI-based weighting"""
        transformer_output, lstm_output, pci_embedded = inputs
        
        # Global average pooling for transformer output
        transformer_pooled = tf.reduce_mean(transformer_output, axis=1)
        
        # Generate fusion weights based on PCI
        fusion_weights = self.pci_weight_generator(pci_embedded)
        transformer_weight = fusion_weights[:, 0:1]
        lstm_weight = fusion_weights[:, 1:2]
        
        # Weighted fusion
        # Ensure dimensions match - project LSTM output to transformer dimension
        lstm_projected = self.lstm_projection(lstm_output)
        
        # Ensure both tensors have compatible shapes for element-wise operations
        # Get the shapes
        transformer_shape = tf.shape(transformer_pooled)
        lstm_shape = tf.shape(lstm_projected)
        
        # Make sure both have the same number of dimensions (2D: batch_size, features)
        if len(transformer_pooled.shape) == 2 and len(lstm_projected.shape) == 2:
            # Both are already 2D, good to go
            pass
        else:
            # Flatten to 2D if needed
            transformer_pooled = tf.reshape(transformer_pooled, [transformer_shape[0], -1])
            lstm_projected = tf.reshape(lstm_projected, [lstm_shape[0], -1])
        
        fused = (transformer_weight * transformer_pooled + 
                lstm_weight * lstm_projected)
        
        # Project to output dimension
        output = self.output_projection(fused)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'transformer_dim': self.transformer_dim,
            'lstm_dim': self.lstm_dim
        })
        return config

class RLWeightAdjustmentLayer(layers.Layer):
    """Reinforcement learning layer for post-match weight adjustment"""
    
    def __init__(self, **kwargs):
        super(RLWeightAdjustmentLayer, self).__init__(**kwargs)
        
        # RL adjustment parameters (trainable)
        self.adjustment_weights = self.add_weight(
            name='rl_adjustment_weights',
            shape=(1,),
            initializer='ones',
            trainable=True
        )
        
        # PCI-based adjustment scaling
        self.pci_scaling = layers.Dense(1, activation='tanh')
        
    def call(self, inputs):
        """Apply RL-based prediction adjustment"""
        prediction, pci_input = inputs
        
        # Calculate PCI-based scaling factor
        pci_scale = self.pci_scaling(pci_input)
        
        # Apply adjustment (subtle modification)
        adjustment = self.adjustment_weights * pci_scale * 0.1  # Small adjustment
        adjusted_prediction = prediction + adjustment
        
        # Ensure output stays in [0, 1] range
        adjusted_prediction = tf.clip_by_value(adjusted_prediction, 0.0, 1.0)
        
        return adjusted_prediction

class PositionalEncoding(layers.Layer):
    """Enhanced positional encoding for transformer"""
    
    def __init__(self, sequence_length: int, d_model: int, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Create positional encoding matrix
        self.pos_encoding = self._create_positional_encoding()
    
    def _create_positional_encoding(self):
        """Create positional encoding matrix"""
        position = np.arange(self.sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pos_encoding = np.zeros((self.sequence_length, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return tf.constant(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        """Add positional encoding to inputs"""
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model
        })
        return config

def create_enhanced_model(
    sequence_length: int = 50,
    feature_dim: int = 64,
    learning_rate: float = 0.001,
    enable_rl_loop: bool = True
) -> EnhancedTransformerLSTMModel:
    """Create and compile the enhanced model with PCI integration"""
    
    model = EnhancedTransformerLSTMModel(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        enable_rl_loop=enable_rl_loop
    )
    
    # Custom loss function with confidence penalty
    def confidence_aware_loss(y_true, y_pred_dict):
        """Loss function that incorporates confidence estimation"""
        prediction = y_pred_dict['prediction']
        confidence = y_pred_dict['confidence']
        
        # Binary crossentropy for prediction
        pred_loss = tf.keras.losses.binary_crossentropy(y_true, prediction)
        
        # Confidence penalty (encourage high confidence for correct predictions)
        correct_predictions = tf.cast(
            tf.equal(tf.round(prediction), y_true), tf.float32
        )
        confidence_penalty = -tf.reduce_mean(
            correct_predictions * tf.math.log(confidence + 1e-8) +
            (1 - correct_predictions) * tf.math.log(1 - confidence + 1e-8)
        )
        
        return pred_loss + 0.1 * confidence_penalty
    
    # Compile model with AdamW optimizer
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01),
        loss=confidence_aware_loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def create_enhanced_callbacks(model_path: str, patience: int = 10):
    """Create enhanced training callbacks with PCI monitoring"""
    
    callbacks = [
        # Early stopping with PCI-aware patience
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint with versioning
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{model_path}/v3.0-PCI-RL_{{epoch:02d}}-{{val_auc:.3f}}.h5",
            monitor='val_auc',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        
        # Adaptive learning rate with PCI consideration
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Enhanced TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/v3.0-PCI-RL',
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            profile_batch='10,20'
        ),
        
        # Custom PCI monitoring callback
        PCIMonitoringCallback()
    ]
    
    return callbacks

class PCIMonitoringCallback(tf.keras.callbacks.Callback):
    """Custom callback to monitor PCI distribution and model performance correlation"""
    
    def __init__(self):
        super().__init__()
        self.pci_history = []
        self.accuracy_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Monitor PCI correlation with model performance"""
        if logs:
            val_accuracy = logs.get('val_accuracy', 0)
            self.accuracy_history.append(val_accuracy)
            
            # Log PCI-related metrics
            logger.info(f"Epoch {epoch + 1}: Val Accuracy = {val_accuracy:.4f}")
            
            # Check for PCI drift (simplified)
            if len(self.accuracy_history) > 5:
                recent_acc = np.mean(self.accuracy_history[-5:])
                older_acc = np.mean(self.accuracy_history[-10:-5]) if len(self.accuracy_history) >= 10 else recent_acc
                
                if abs(recent_acc - older_acc) > 0.05:  # 5% accuracy drift
                    logger.warning(f"Potential PCI drift detected: accuracy change = {recent_acc - older_acc:.4f}")
