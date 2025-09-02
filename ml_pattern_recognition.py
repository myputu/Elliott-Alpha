"""
Machine Learning Pattern Recognition System for Elliott Wave Trading
This module implements advanced machine learning algorithms for improved
Elliott Wave pattern recognition and trading signal generation.

Author: Manus AI
Date: 28 Agustus 2025
Version: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import threading
import time

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PatternFeatures:
    """Elliott Wave pattern features for ML training."""
    price_ratios: List[float]  # Fibonacci ratios between waves
    time_ratios: List[float]   # Time relationships between waves
    volume_profile: List[float]  # Volume characteristics
    momentum_indicators: List[float]  # RSI, MACD, etc.
    wave_angles: List[float]   # Angles of wave movements
    retracement_levels: List[float]  # Fibonacci retracement levels
    pattern_type: str  # Target pattern classification
    confidence_score: float  # Pattern confidence (0-1)

@dataclass
class MLPrediction:
    """Machine learning prediction result."""
    pattern_type: str
    confidence: float
    probability_distribution: Dict[str, float]
    feature_importance: Dict[str, float]
    timestamp: datetime

class ElliotWaveMLClassifier:
    """
    Advanced machine learning classifier for Elliott Wave patterns
    using ensemble methods and neural networks.
    """
    
    def __init__(self, model_type: str = "ensemble"):
        """
        Initialize ML classifier.
        
        Args:
            model_type: Type of ML model ("ensemble", "neural_network", "gradient_boost")
        """
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.pattern_classes = [
            "impulse_wave_1", "impulse_wave_2", "impulse_wave_3", 
            "impulse_wave_4", "impulse_wave_5",
            "corrective_wave_a", "corrective_wave_b", "corrective_wave_c",
            "triangle", "flat", "zigzag", "complex_correction",
            "no_pattern"
        ]
        self.is_trained = False
        self.training_history = []
        self._lock = threading.Lock()
        
        logger.info(f"Elliott Wave ML Classifier initialized with {model_type} model")
    
    def _create_models(self):
        """Create ML models based on specified type."""
        if self.model_type == "ensemble":
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.models['gradient_boost'] = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
            
        elif self.model_type == "neural_network":
            self.models['mlp'] = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
            
        elif self.model_type == "gradient_boost":
            self.models['xgboost'] = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=10,
                subsample=0.8,
                random_state=42
            )
        
        # Create scalers for each model
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def extract_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame = None) -> np.ndarray:
        """
        Extract features from price and volume data for ML training/prediction.
        
        Args:
            price_data: DataFrame with OHLC data
            volume_data: Optional volume data
            
        Returns:
            Feature array for ML models
        """
        features = []
        
        # Price-based features
        if len(price_data) >= 5:
            # Wave ratios (Fibonacci relationships)
            highs = price_data['high'].values
            lows = price_data['low'].values
            closes = price_data['close'].values
            
            # Calculate wave measurements
            wave_heights = []
            for i in range(1, len(highs)):
                wave_heights.append(abs(highs[i] - lows[i-1]))
            
            # Fibonacci ratios between waves
            if len(wave_heights) >= 3:
                ratio_1_2 = wave_heights[1] / wave_heights[0] if wave_heights[0] != 0 else 0
                ratio_2_3 = wave_heights[2] / wave_heights[1] if wave_heights[1] != 0 else 0
                features.extend([ratio_1_2, ratio_2_3])
            else:
                features.extend([0, 0])
            
            # Time ratios
            time_diffs = np.diff(price_data.index.astype(np.int64))
            if len(time_diffs) >= 2:
                time_ratio = time_diffs[1] / time_diffs[0] if time_diffs[0] != 0 else 0
                features.append(time_ratio)
            else:
                features.append(0)
            
            # Price momentum features
            price_changes = np.diff(closes)
            if len(price_changes) >= 3:
                momentum_1 = price_changes[-1]
                momentum_2 = price_changes[-2]
                momentum_3 = price_changes[-3]
                features.extend([momentum_1, momentum_2, momentum_3])
            else:
                features.extend([0, 0, 0])
            
            # Volatility features
            volatility = np.std(price_changes) if len(price_changes) > 1 else 0
            features.append(volatility)
            
            # Retracement levels
            if len(closes) >= 3:
                recent_high = max(closes[-3:])
                recent_low = min(closes[-3:])
                current_price = closes[-1]
                
                if recent_high != recent_low:
                    retracement = (recent_high - current_price) / (recent_high - recent_low)
                    features.append(retracement)
                else:
                    features.append(0)
            else:
                features.append(0)
            
            # Wave angles (simplified)
            if len(closes) >= 2:
                angle = np.arctan2(closes[-1] - closes[-2], 1) * 180 / np.pi
                features.append(angle)
            else:
                features.append(0)
        
        # Volume features (if available)
        if volume_data is not None and len(volume_data) >= 3:
            volumes = volume_data['volume'].values
            volume_trend = (volumes[-1] - volumes[-3]) / volumes[-3] if volumes[-3] != 0 else 0
            avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else np.mean(volumes)
            volume_ratio = volumes[-1] / avg_volume if avg_volume != 0 else 0
            features.extend([volume_trend, volume_ratio])
        else:
            features.extend([0, 0])
        
        # Technical indicators (simplified)
        if len(closes) >= 14:
            # RSI-like indicator
            gains = np.maximum(price_changes, 0)
            losses = np.maximum(-price_changes, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss != 0 else 50
            features.append(rsi)
        else:
            features.append(50)  # Neutral RSI
        
        # Ensure consistent feature length
        target_length = 15  # Expected number of features
        while len(features) < target_length:
            features.append(0)
        
        features = features[:target_length]  # Truncate if too long
        
        return np.array(features)
    
    def prepare_training_data(self, historical_patterns: List[PatternFeatures]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from historical patterns.
        
        Args:
            historical_patterns: List of labeled pattern features
            
        Returns:
            Tuple of (features, labels)
        """
        X = []
        y = []
        
        for pattern in historical_patterns:
            # Combine all feature types into single feature vector
            feature_vector = []
            feature_vector.extend(pattern.price_ratios[:3])  # First 3 price ratios
            feature_vector.extend(pattern.time_ratios[:2])   # First 2 time ratios
            feature_vector.extend(pattern.volume_profile[:2]) # First 2 volume features
            feature_vector.extend(pattern.momentum_indicators[:3]) # First 3 momentum indicators
            feature_vector.extend(pattern.wave_angles[:2])   # First 2 wave angles
            feature_vector.extend(pattern.retracement_levels[:3]) # First 3 retracement levels
            
            # Ensure consistent length
            while len(feature_vector) < 15:
                feature_vector.append(0)
            feature_vector = feature_vector[:15]
            
            X.append(feature_vector)
            y.append(pattern.pattern_type)
        
        return np.array(X), np.array(y)
    
    def train(self, training_patterns: List[PatternFeatures], validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the ML models on historical Elliott Wave patterns.
        
        Args:
            training_patterns: List of labeled pattern features
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Training ML models with {len(training_patterns)} patterns")
        
        # Prepare training data
        X, y = self.prepare_training_data(training_patterns)
        
        if len(X) == 0:
            raise ValueError("No training data available")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Create models
        self._create_models()
        
        training_results = {}
        
        with self._lock:
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name}...")
                
                # Scale features
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)
                
                # Train model
                start_time = time.time()
                model.fit(X_train_scaled, y_train)
                training_time = time.time() - start_time
                
                # Evaluate model
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                
                # Predictions for detailed metrics
                y_pred = model.predict(X_test_scaled)
                
                training_results[model_name] = {
                    'training_time': training_time,
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                logger.info(f"{model_name} - Train: {train_score:.3f}, Test: {test_score:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
            self.is_trained = True
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Store training history
            self.training_history.append({
                'timestamp': datetime.now(),
                'num_patterns': len(training_patterns),
                'results': training_results
            })
        
        logger.info("ML model training completed")
        return training_results
    
    def predict(self, price_data: pd.DataFrame, volume_data: pd.DataFrame = None) -> MLPrediction:
        """
        Predict Elliott Wave pattern from price data.
        
        Args:
            price_data: DataFrame with OHLC data
            volume_data: Optional volume data
            
        Returns:
            ML prediction result
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        # Extract features
        features = self.extract_features(price_data, volume_data)
        features = features.reshape(1, -1)
        
        predictions = {}
        probabilities = {}
        
        with self._lock:
            for model_name, model in self.models.items():
                # Scale features
                features_scaled = self.scalers[model_name].transform(features)
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                predictions[model_name] = prediction
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0]
                    probabilities[model_name] = dict(zip(model.classes_, proba))
        
        # Ensemble prediction (majority vote or average probabilities)
        if len(predictions) > 1:
            # Use probability averaging for ensemble
            ensemble_probs = {}
            for pattern_class in self.pattern_classes:
                probs = []
                for model_name in probabilities:
                    if pattern_class in probabilities[model_name]:
                        probs.append(probabilities[model_name][pattern_class])
                
                if probs:
                    ensemble_probs[pattern_class] = np.mean(probs)
                else:
                    ensemble_probs[pattern_class] = 0.0
            
            # Get best prediction
            best_pattern = max(ensemble_probs, key=ensemble_probs.get)
            confidence = ensemble_probs[best_pattern]
            
        else:
            # Single model prediction
            model_name = list(predictions.keys())[0]
            best_pattern = predictions[model_name]
            ensemble_probs = probabilities.get(model_name, {})
            confidence = ensemble_probs.get(best_pattern, 0.5)
        
        # Feature importance (from Random Forest if available)
        feature_importance = {}
        if 'random_forest' in self.models:
            importances = self.models['random_forest'].feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
        
        return MLPrediction(
            pattern_type=best_pattern,
            confidence=confidence,
            probability_distribution=ensemble_probs,
            feature_importance=feature_importance,
            timestamp=datetime.now()
        )
    
    def save_models(self, filepath: str):
        """Save trained models to file."""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'pattern_classes': self.pattern_classes,
            'model_type': self.model_type,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from file."""
        model_data = joblib.load(filepath)
        
        with self._lock:
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']
            self.pattern_classes = model_data['pattern_classes']
            self.model_type = model_data['model_type']
            self.training_history = model_data.get('training_history', [])
            self.is_trained = True
        
        logger.info(f"Models loaded from {filepath}")

class PatternDataGenerator:
    """
    Generate synthetic Elliott Wave pattern data for ML training.
    """
    
    def __init__(self):
        self.fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
        
    def generate_impulse_wave_patterns(self, num_patterns: int = 100) -> List[PatternFeatures]:
        """Generate synthetic impulse wave patterns."""
        patterns = []
        
        for _ in range(num_patterns):
            # Generate realistic Elliott Wave ratios
            wave_1_height = np.random.uniform(0.5, 2.0)
            wave_2_retracement = np.random.choice([0.382, 0.5, 0.618])
            wave_3_extension = np.random.choice([1.618, 2.618, 1.0])
            wave_4_retracement = np.random.choice([0.236, 0.382, 0.5])
            wave_5_ratio = np.random.choice([0.618, 1.0, 1.618])
            
            # Price ratios
            price_ratios = [
                wave_3_extension,  # Wave 3 to Wave 1 ratio
                wave_5_ratio,      # Wave 5 to Wave 1 ratio
                wave_2_retracement, # Wave 2 retracement
                wave_4_retracement  # Wave 4 retracement
            ]
            
            # Time ratios (waves often have time relationships)
            time_ratios = [
                np.random.uniform(0.5, 2.0),  # Time ratio between waves
                np.random.uniform(0.8, 1.5)
            ]
            
            # Volume profile (typically increases in wave 3)
            volume_profile = [
                np.random.uniform(0.8, 1.2),  # Wave 1 volume
                np.random.uniform(1.2, 2.0)   # Wave 3 volume increase
            ]
            
            # Momentum indicators
            momentum_indicators = [
                np.random.uniform(30, 70),     # RSI-like
                np.random.uniform(-0.5, 0.5),  # MACD-like
                np.random.uniform(0.5, 1.5)    # Momentum
            ]
            
            # Wave angles
            wave_angles = [
                np.random.uniform(30, 60),     # Steep upward angle
                np.random.uniform(-30, -10)    # Correction angle
            ]
            
            # Retracement levels
            retracement_levels = [
                wave_2_retracement,
                wave_4_retracement,
                np.random.choice(self.fibonacci_ratios)
            ]
            
            # Determine specific wave type
            wave_types = ["impulse_wave_1", "impulse_wave_3", "impulse_wave_5"]
            pattern_type = np.random.choice(wave_types)
            
            pattern = PatternFeatures(
                price_ratios=price_ratios,
                time_ratios=time_ratios,
                volume_profile=volume_profile,
                momentum_indicators=momentum_indicators,
                wave_angles=wave_angles,
                retracement_levels=retracement_levels,
                pattern_type=pattern_type,
                confidence_score=np.random.uniform(0.7, 0.95)
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def generate_corrective_wave_patterns(self, num_patterns: int = 100) -> List[PatternFeatures]:
        """Generate synthetic corrective wave patterns."""
        patterns = []
        
        for _ in range(num_patterns):
            # Corrective waves have different characteristics
            correction_depth = np.random.choice([0.382, 0.5, 0.618, 0.786])
            
            # Price ratios for corrections
            price_ratios = [
                correction_depth,              # Correction depth
                np.random.uniform(0.5, 1.5),   # Wave relationships
                np.random.uniform(0.8, 1.2),
                np.random.uniform(0.6, 1.4)
            ]
            
            # Time ratios (corrections often take time)
            time_ratios = [
                np.random.uniform(1.0, 3.0),   # Corrections can be time-consuming
                np.random.uniform(0.5, 2.0)
            ]
            
            # Volume profile (typically decreases in corrections)
            volume_profile = [
                np.random.uniform(0.5, 0.8),   # Lower volume
                np.random.uniform(0.3, 0.7)
            ]
            
            # Momentum indicators (often oversold/overbought)
            momentum_indicators = [
                np.random.choice([np.random.uniform(20, 35), np.random.uniform(65, 80)]),  # Extreme RSI
                np.random.uniform(-1.0, 1.0),   # MACD
                np.random.uniform(0.2, 0.8)     # Lower momentum
            ]
            
            # Wave angles (corrections often sideways or against trend)
            wave_angles = [
                np.random.uniform(-45, 45),     # Variable angles
                np.random.uniform(-60, 60)
            ]
            
            # Retracement levels
            retracement_levels = [
                correction_depth,
                np.random.choice(self.fibonacci_ratios),
                np.random.choice(self.fibonacci_ratios)
            ]
            
            # Corrective wave types
            corrective_types = ["corrective_wave_a", "corrective_wave_b", "corrective_wave_c", 
                              "triangle", "flat", "zigzag"]
            pattern_type = np.random.choice(corrective_types)
            
            pattern = PatternFeatures(
                price_ratios=price_ratios,
                time_ratios=time_ratios,
                volume_profile=volume_profile,
                momentum_indicators=momentum_indicators,
                wave_angles=wave_angles,
                retracement_levels=retracement_levels,
                pattern_type=pattern_type,
                confidence_score=np.random.uniform(0.6, 0.9)
            )
            
            patterns.append(pattern)
        
        return patterns

def setup_ml_pattern_recognition(model_type: str = "ensemble") -> ElliotWaveMLClassifier:
    """
    Setup ML pattern recognition system with training data.
    
    Args:
        model_type: Type of ML model to use
        
    Returns:
        Trained ML classifier
    """
    logger.info("Setting up ML pattern recognition system...")
    
    # Create classifier
    classifier = ElliotWaveMLClassifier(model_type=model_type)
    
    # Generate training data
    data_generator = PatternDataGenerator()
    
    # Generate synthetic patterns for training
    impulse_patterns = data_generator.generate_impulse_wave_patterns(200)
    corrective_patterns = data_generator.generate_corrective_wave_patterns(200)
    
    # Combine all patterns
    all_patterns = impulse_patterns + corrective_patterns
    
    # Add some "no pattern" examples
    no_pattern_examples = []
    for _ in range(50):
        # Random features that don't follow Elliott Wave rules
        pattern = PatternFeatures(
            price_ratios=[np.random.uniform(0, 3) for _ in range(4)],
            time_ratios=[np.random.uniform(0, 3) for _ in range(2)],
            volume_profile=[np.random.uniform(0, 2) for _ in range(2)],
            momentum_indicators=[np.random.uniform(0, 100), np.random.uniform(-2, 2), np.random.uniform(0, 2)],
            wave_angles=[np.random.uniform(-90, 90) for _ in range(2)],
            retracement_levels=[np.random.uniform(0, 2) for _ in range(3)],
            pattern_type="no_pattern",
            confidence_score=np.random.uniform(0.3, 0.7)
        )
        no_pattern_examples.append(pattern)
    
    all_patterns.extend(no_pattern_examples)
    
    # Train the classifier
    training_results = classifier.train(all_patterns)
    
    logger.info("ML pattern recognition system setup completed")
    logger.info(f"Training results: {training_results}")
    
    return classifier

if __name__ == "__main__":
    # Example usage and testing
    print("Testing ML Pattern Recognition...")
    
    # Setup ML system
    ml_classifier = setup_ml_pattern_recognition("ensemble")
    
    # Create sample price data for testing
    dates = pd.date_range(start='2024-01-01', periods=20, freq='H')
    sample_prices = pd.DataFrame({
        'open': np.random.uniform(1900, 2100, 20),
        'high': np.random.uniform(1950, 2150, 20),
        'low': np.random.uniform(1850, 2050, 20),
        'close': np.random.uniform(1900, 2100, 20)
    }, index=dates)
    
    # Make prediction
    prediction = ml_classifier.predict(sample_prices)
    
    print(f"Prediction: {prediction.pattern_type}")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"Probabilities: {prediction.probability_distribution}")
    
    # Save models
    ml_classifier.save_models("elliott_wave_ml_models.joblib")
    print("Models saved successfully")
    
    print("ML Pattern Recognition test completed")

