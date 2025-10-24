import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TransformerLSTMModel(tf.keras.Model):
    """Hybrid Transformer-LSTM model for win prediction"""
    
    def __init__(
        self,
        sequence_length: int = 50,
        feature_dim: int = 64,
        d_model: int = 128,
        num_heads: int = 8,
        num_transformer_layers: int = 4,
        lstm_units: int = 64,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super(TransformerLSTMModel, self).__init__(**kwargs)
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        # Input projection
        self.input_projection = layers.Dense(d_model, activation='relu')
        self.input_dropout = layers.Dropout(dropout_rate)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(sequence_length, d_model)
        
        # Transformer layers
        self.transformer_layers = []
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(
                TransformerBlock(d_model, num_heads, dropout_rate)
            )
        
        # LSTM layers
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)
        self.lstm2 = layers.LSTM(lstm_units // 2, return_sequences=False, dropout=dropout_rate)
        
        # Output layers
        self.dense1 = layers.Dense(64, activation='relu')
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(32, activation='relu')
        self.dropout2 = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(1, activation='sigmoid')
        
        # Batch normalization
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()
    
    def call(self, inputs, training=None, mask=None):
        """Forward pass"""
        # Input shape: (batch_size, sequence_length, feature_dim)
        
        # Project input features to model dimension
        x = self.input_projection(inputs)
        x = self.input_dropout(x, training=training)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training=training)
        
        # Apply LSTM layers
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        
        # Apply dense layers with batch normalization
        x = self.dense1(x)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)
        
        # Output prediction
        output = self.output_layer(x)
        
        return output
    
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
            'dropout_rate': self.dropout_rate
        })
        return config

class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention and feed-forward network"""
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Multi-head attention
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
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
        """Forward pass through transformer block"""
        
        # Multi-head attention with residual connection
        attn_output = self.multi_head_attention(
            inputs, inputs, attention_mask=mask, training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        """Get layer configuration"""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config

class PositionalEncoding(layers.Layer):
    """Positional encoding for transformer"""
    
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
        """Get layer configuration"""
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model
        })
        return config

def create_model(
    sequence_length: int = 50,
    feature_dim: int = 64,
    learning_rate: float = 0.001
) -> TransformerLSTMModel:
    """Create and compile the model"""
    
    model = TransformerLSTMModel(
        sequence_length=sequence_length,
        feature_dim=feature_dim
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def create_callbacks(model_path: str, patience: int = 10):
    """Create training callbacks"""
    
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_auc',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    
    return callbacks

class ModelEvaluator:
    """Utility class for model evaluation"""
    
    @staticmethod
    def evaluate_model(model: TransformerLSTMModel, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model performance"""
        
        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics, y_pred_proba
    
    @staticmethod
    def plot_training_history(history):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # AUC
        axes[1, 1].plot(history.history['auc'], label='Training AUC')
        axes[1, 1].plot(history.history['val_auc'], label='Validation AUC')
        axes[1, 1].set_title('Model AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        plt.show()
