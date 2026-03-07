"""
Frequency-Based Matching Pipeline
Predicts categorical feature values based on training frequency
"""

import pandas as pd
import pickle
from typing import Dict, List, Tuple


class FrequencyMatcher:
    """Predicts categorical features using frequency-based matching"""
    
    def __init__(
        self,
        lookup_path: str = 'lookup.pkl',
        taxonomy_path: str = 'taxonomy/taxonomy.parquet',
        frequency_threshold: float = 0.0
    ):
        """
        Initialize matcher
        
        Args:
            lookup_path: Path to lookup table pickle
            taxonomy_path: Path to taxonomy parquet
            frequency_threshold: Minimum frequency % to make prediction (0-100)
        """
        self.frequency_threshold = frequency_threshold
        
        print(f"Loading lookup table from {lookup_path}...")
        with open(lookup_path, 'rb') as f:
            self.lookup = pickle.load(f)
        
        print(f"Loading taxonomy from {taxonomy_path}...")
        self.taxonomy = pd.read_parquet(taxonomy_path)
        
        # Build category -> feature_name -> feature_type mapping
        self.category_features = {}
        for _, row in self.taxonomy.iterrows():
            category = row['category']
            feature_name = row['feature_name']
            feature_type = row['feature_type']
            
            if category not in self.category_features:
                self.category_features[category] = {}
            
            self.category_features[category][feature_name] = feature_type
        
        print(f"✓ Loaded {len(self.lookup)} categories in lookup")
        print(f"✓ Loaded {len(self.category_features)} categories in taxonomy")
        print(f"✓ Frequency threshold: {self.frequency_threshold}%")
    
    def predict_for_product(self, category: str) -> List[Tuple[str, str]]:
        """
        Predict categorical features for a product category
        
        Args:
            category: Product category
            
        Returns:
            List of (feature_name, predicted_value) tuples
        """
        predictions = []
        
        # Get expected features for this category from taxonomy
        if category not in self.category_features:
            return predictions
        
        # For each categorical feature in this category
        for feature_name, feature_type in self.category_features[category].items():
            if feature_type != 'categorical':
                continue
            
            # Look up in frequency table
            if category not in self.lookup:
                continue
            
            if feature_name not in self.lookup[category]:
                continue
            
            # Get most frequent value
            values = self.lookup[category][feature_name]
            if not values:
                continue
            
            # Find top value
            top_value = max(values.items(), key=lambda x: x[1])
            value, frequency = top_value
            
            # Only predict if above threshold
            if frequency >= self.frequency_threshold:
                predictions.append((feature_name, value))
        
        return predictions
    
    def evaluate(
        self,
        val_products_path: str = 'val_reshuffled/products.parquet',
        val_features_path: str = 'val_reshuffled/product_features.parquet'
    ) -> Dict:
        """
        Evaluate frequency matching on validation set
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*80)
        print("FREQUENCY MATCHING EVALUATION")
        print("="*80)
        print(f"Frequency threshold: {self.frequency_threshold}%")
        
        # Load validation data
        print("\nLoading validation data...")
        val_products = pd.read_parquet(val_products_path)
        val_features = pd.read_parquet(val_features_path)
        
        # Filter to categorical only
        val_categorical = val_features[val_features['feature_type'] == 'categorical']
        
        print(f"Validation products: {len(val_products)}")
        print(f"Validation categorical features: {len(val_categorical)}")
        
        # Make predictions for each product
        print("\nMaking predictions...")
        all_predictions = {}
        
        for _, product in val_products.iterrows():
            uid = product['uid']
            category = product['category']
            
            predictions = self.predict_for_product(category)
            all_predictions[uid] = {feat: val for feat, val in predictions}
        
        # Evaluate against ground truth
        print("Evaluating predictions...")
        
        total_ground_truth = 0
        total_predictions = 0
        correct_predictions = 0
        missing_predictions = 0
        wrong_predictions = 0
        
        for _, row in val_categorical.iterrows():
            uid = row['uid']
            feature_name = row['feature_name']
            ground_truth_value = row['feature_value']
            
            total_ground_truth += 1
            
            # Check if we made a prediction for this (uid, feature_name)
            if uid in all_predictions and feature_name in all_predictions[uid]:
                predicted_value = all_predictions[uid][feature_name]
                total_predictions += 1
                
                # Check if correct
                if predicted_value == ground_truth_value:
                    correct_predictions += 1
                else:
                    wrong_predictions += 1
            else:
                missing_predictions += 1
        
        # Calculate metrics
        accuracy = (correct_predictions / total_ground_truth * 100) if total_ground_truth > 0 else 0
        precision = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        recall = (correct_predictions / total_ground_truth * 100) if total_ground_truth > 0 else 0
        
        # Results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"\nGround truth instances: {total_ground_truth}")
        print(f"Predictions made: {total_predictions}")
        print(f"Missing predictions: {missing_predictions}")
        print(f"\nCorrect predictions: {correct_predictions}")
        print(f"Wrong predictions: {wrong_predictions}")
        print(f"\nExact Match Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"Recall: {recall:.2f}%")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'total_ground_truth': total_ground_truth,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'wrong_predictions': wrong_predictions,
            'missing_predictions': missing_predictions
        }


def main():
    """Main execution"""
    # Test with different thresholds
    thresholds = [60.0, 70.0, 80.0, 85.0, 90.0, 95.0]
    
    results = []
    for threshold in thresholds:
        print("\n" + "="*80)
        print(f"TESTING THRESHOLD: {threshold}%")
        print("="*80)
        
        matcher = FrequencyMatcher(frequency_threshold=threshold)
        metrics = matcher.evaluate()
        results.append((threshold, metrics))
    
    # Summary
    print("\n" + "="*80)
    print("THRESHOLD COMPARISON")
    print("="*80)
    print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'Predictions':<12}")
    print("-"*80)
    for threshold, metrics in results:
        print(f"{threshold:<12.1f} {metrics['accuracy']:<12.2f} {metrics['precision']:<12.2f} {metrics['recall']:<12.2f} {metrics['total_predictions']:<12}")


if __name__ == '__main__':
    main()
