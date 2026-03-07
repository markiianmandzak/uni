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
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Test with different thresholds (50 to 95 with step 5)
    thresholds = list(range(0, 100, 5))
    
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
    
    # Plot results
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION")
    print("="*80)
    
    thresholds_list = [r[0] for r in results]
    accuracies = [r[1]['accuracy'] for r in results]
    precisions = [r[1]['precision'] for r in results]
    
    # Calculate F1 score for each threshold (harmonic mean of precision and accuracy)
    f1_scores = []
    for p, a in zip(precisions, accuracies):
        if p + a > 0:
            f1 = 2 * (p * a) / (p + a)
        else:
            f1 = 0
        f1_scores.append(f1)
    
    # Find optimal threshold (max F1 score)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_list[optimal_idx]
    optimal_accuracy = accuracies[optimal_idx]
    optimal_precision = precisions[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot accuracy and precision
    ax.plot(thresholds_list, accuracies, 'o-', label='Accuracy (Recall)', linewidth=2.5, markersize=8, color='#2E86AB')
    ax.plot(thresholds_list, precisions, 's-', label='Precision', linewidth=2.5, markersize=8, color='#A23B72')
    ax.plot(thresholds_list, f1_scores, '^-', label='F1 Score', linewidth=2.5, markersize=8, color='#F18F01')
    
    # Draw vertical line at optimal threshold
    ax.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal ({optimal_threshold}%)')
    
    # Add text annotation at optimal point
    ax.text(optimal_threshold + 2, optimal_f1 - 5, 
            f'Optimal: {optimal_threshold}%\nF1: {optimal_f1:.1f}%\nAcc: {optimal_accuracy:.1f}%\nPrec: {optimal_precision:.1f}%',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    ax.set_xlabel('Frequency Threshold (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Frequency Matching: Finding Optimal Threshold via F1 Score', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-5, 100)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('frequency_matching_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot to frequency_matching_results.png")
    
    # Report optimal threshold
    print(f"\n{'='*80}")
    print(f"OPTIMAL THRESHOLD (Max F1 Score)")
    print(f"{'='*80}")
    print(f"Threshold: {optimal_threshold}%")
    print(f"  F1 Score: {optimal_f1:.2f}%")
    print(f"  Accuracy: {optimal_accuracy:.2f}%")
    print(f"  Precision: {optimal_precision:.2f}%")
    print(f"  Predictions: {results[optimal_idx][1]['total_predictions']}")
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION: Use {optimal_threshold}% threshold for best balance")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
