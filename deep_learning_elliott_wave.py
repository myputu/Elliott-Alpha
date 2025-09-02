import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, Attention
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DeepLearningElliottWave:
    """
    Deep Learning module for Elliott Wave pattern recognition using LSTM and CNN architectures.
    Based on research findings from neural network applications to Elliott Wave analysis.
    """
    
    def __init__(self, sequence_length=60, n_features=5):
        """
        Initialize the deep learning Elliott Wave analyzer.
        
        Args:
            sequence_length (int): Length of input sequences for time series analysis
            n_features (int): Number of features (OHLCV)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.scaler = MinMaxScaler()
        self.model = None
        self.pattern_classifier = None
        
    def prepare_data(self, ohlc_data):
        """
        Prepare OHLC data for deep learning model training.
        
        Args:
            ohlc_data (pd.DataFrame): OHLC data with columns [open, high, low, close, volume]
            
        Returns:
            tuple: (X, y) prepared sequences and labels
        """
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in ohlc_data.columns for col in required_cols):
            # If volume is missing, create a dummy volume column
            if 'volume' not in ohlc_data.columns:
                ohlc_data['volume'] = 1000  # Dummy volume
        
        # Select features
        features = ohlc_data[required_cols].values
        
        # Normalize the data
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            # For now, create dummy labels (0: no pattern, 1: bullish pattern, 2: bearish pattern)
            # In a real implementation, these would be manually labeled or derived from expert analysis
            y.append(np.random.randint(0, 3))  # Placeholder labels
            
        return np.array(X), np.array(y)
    
    def create_lstm_model(self):
        """
        Create an LSTM-based model for Elliott Wave pattern recognition.
        Based on research showing LSTM effectiveness for time series pattern recognition.
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes: no pattern, bullish, bearish
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def create_cnn_lstm_model(self):
        """
        Create a hybrid CNN-LSTM model for Elliott Wave pattern recognition.
        CNN layers extract local patterns, LSTM captures temporal dependencies.
        """
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                   input_shape=(self.sequence_length, self.n_features)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def create_attention_model(self):
        """
        Create an attention-based model for Elliott Wave pattern recognition.
        Attention mechanism helps focus on important parts of the sequence.
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layer
        lstm_out = LSTM(50, return_sequences=True)(inputs)
        lstm_out = Dropout(0.2)(lstm_out)
        
        # Attention mechanism (simplified)
        attention_weights = Dense(1, activation='tanh')(lstm_out)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        context_vector = tf.reduce_sum(lstm_out * attention_weights, axis=1)
        
        # Output layers
        dense_out = Dense(25, activation='relu')(context_vector)
        outputs = Dense(3, activation='softmax')(dense_out)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def train_model(self, X, y, model_type='lstm', epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the deep learning model for Elliott Wave pattern recognition.
        
        Args:
            X (np.array): Input sequences
            y (np.array): Target labels
            model_type (str): Type of model ('lstm', 'cnn_lstm', 'attention')
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            History: Training history
        """
        # Create model based on type
        if model_type == 'lstm':
            self.model = self.create_lstm_model()
        elif model_type == 'cnn_lstm':
            self.model = self.create_cnn_lstm_model()
        elif model_type == 'attention':
            self.model = self.create_attention_model()
        else:
            raise ValueError("Invalid model_type. Choose from 'lstm', 'cnn_lstm', 'attention'")
        
        print(f"Training {model_type} model...")
        print(self.model.summary())
        
        # Train the model
        history = self.model.fit(X, y, 
                               epochs=epochs, 
                               batch_size=batch_size,
                               validation_split=validation_split,
                               verbose=1)
        
        return history
    
    def predict_patterns(self, X):
        """
        Predict Elliott Wave patterns using the trained model.
        
        Args:
            X (np.array): Input sequences
            
        Returns:
            np.array: Predicted pattern probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def create_pattern_features(self, ohlc_data):
        """
        Create additional features for Elliott Wave pattern recognition.
        Based on technical analysis and Elliott Wave theory.
        
        Args:
            ohlc_data (pd.DataFrame): OHLC data
            
        Returns:
            pd.DataFrame: Enhanced data with additional features
        """
        df = ohlc_data.copy()
        
        # Price-based features
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'price_ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
        
        # Volatility features
        df['volatility'] = df['close'].rolling(window=20).std()
        df['atr'] = df['price_range'].rolling(window=14).mean()  # Simplified ATR
        
        # Momentum features
        df['roc_5'] = df['close'].pct_change(5)
        df['roc_10'] = df['close'].pct_change(10)
        
        # Elliott Wave specific features
        df['swing_high'] = df['high'].rolling(window=5, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(window=5, center=True).min() == df['low']
        
        # Fibonacci retracement levels (simplified)
        df['fib_382'] = df['close'].rolling(window=20).min() + 0.382 * (df['close'].rolling(window=20).max() - df['close'].rolling(window=20).min())
        df['fib_618'] = df['close'].rolling(window=20).min() + 0.618 * (df['close'].rolling(window=20).max() - df['close'].rolling(window=20).min())
        
        return df.fillna(method='ffill').fillna(method='bfill')
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test (np.array): Test input sequences
            y_test (np.array): Test target labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Get predictions
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == y_test)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, predicted_classes)
        report = classification_report(y_test, predicted_classes, 
                                     target_names=['No Pattern', 'Bullish', 'Bearish'])
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions,
            'predicted_classes': predicted_classes
        }
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Create dummy OHLC data for testing
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=1000, freq='H')
    
    # Generate synthetic OHLC data
    close_prices = 100 + np.cumsum(np.random.randn(1000) * 0.1)
    high_prices = close_prices + np.abs(np.random.randn(1000) * 0.5)
    low_prices = close_prices - np.abs(np.random.randn(1000) * 0.5)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    volumes = np.random.randint(1000, 10000, 1000)
    
    dummy_data = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # Initialize the deep learning Elliott Wave analyzer
    dl_ew = DeepLearningElliottWave(sequence_length=60, n_features=5)
    
    # Prepare data
    X, y = dl_ew.prepare_data(dummy_data)
    print(f"Prepared data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train LSTM model
    print("Training LSTM model...")
    history = dl_ew.train_model(X_train, y_train, model_type='lstm', epochs=10, batch_size=32)
    
    # Evaluate model
    print("Evaluating model...")
    evaluation = dl_ew.evaluate_model(X_test, y_test)
    print(f"Test Accuracy: {evaluation['accuracy']:.4f}")
    print("Classification Report:")
    print(evaluation['classification_report'])
    
    print("Deep Learning Elliott Wave module test completed.")

