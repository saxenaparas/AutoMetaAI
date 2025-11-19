import pandas as pd
import os
from collections import defaultdict, Counter

class PredictionVerifier:
    def __init__(self):
        self.stats = defaultdict(Counter)
        self.missing_patterns = defaultdict(Counter)
        self.incorrect_predictions = []
    
    def verify_predictions(self, predicted_file, verified_file):
        """Compare predicted file against verified ground truth"""
        print(f"🔍 Comparing predictions against verified data...")
        print(f"   Predicted: {predicted_file}")
        print(f"   Verified:  {verified_file}")
        
        # Read both files
        pred_df = pd.read_excel(predicted_file)
        true_df = pd.read_excel(verified_file)
        
        # Ensure both have same rows (merge on dataTagId)
        merged = pd.merge(pred_df, true_df, on='dataTagId', suffixes=('_pred', '_true'))
        
        print(f"📊 Comparing {len(merged)} matching rows")
        
        # Columns to compare (all except dataTagId and description)
        columns_to_compare = [col for col in pred_df.columns if col not in ['dataTagId', 'description']]
        
        total_cells = 0
        correct_cells = 0
        
        # Compare each row and column
        for idx, row in merged.iterrows():
            description = row['description'] if 'description' in row else ''
            
            for col in columns_to_compare:
                pred_col = f"{col}_pred"
                true_col = f"{col}_true"
                
                if pred_col in row and true_col in row:
                    total_cells += 1
                    pred_val = row[pred_col]
                    true_val = row[true_col]
                    
                    # Check if prediction is correct
                    if pd.isna(pred_val) and pd.isna(true_val):
                        correct_cells += 1
                        self.stats[col]['correct_empty'] += 1
                    elif pred_val == true_val:
                        correct_cells += 1
                        self.stats[col]['correct'] += 1
                    else:
                        # Incorrect prediction
                        if pd.isna(pred_val):
                            self.stats[col]['missing'] += 1
                            self._analyze_missing_pattern(description, col, true_val)
                        else:
                            self.stats[col]['incorrect'] += 1
                            self.incorrect_predictions.append({
                                'dataTagId': row['dataTagId'],
                                'description': description,
                                'column': col,
                                'predicted': pred_val,
                                'actual': true_val
                            })
        
        # Calculate accuracy
        accuracy = (correct_cells / total_cells * 100) if total_cells > 0 else 0
        print(f"\n🎯 OVERALL ACCURACY: {accuracy:.1f}% ({correct_cells}/{total_cells} cells)")
        
        self._print_detailed_report()
        self._print_missing_patterns()
        self._print_incorrect_predictions()
        
        return accuracy
    
    def _analyze_missing_pattern(self, description, column, true_value):
        """Analyze patterns that were missed"""
        words = str(description).upper().split()
        for word in words:
            self.missing_patterns[column][(word, true_value)] += 1
    
    def _print_detailed_report(self):
        """Print column-by-column accuracy"""
        print(f"\n📈 COLUMN-WISE ACCURACY:")
        print(f"{'Column':<25} {'Correct':<8} {'Missing':<8} {'Wrong':<8} {'Accuracy':<10}")
        print("-" * 65)
        
        for col, counts in self.stats.items():
            total = sum(counts.values())
            correct = counts['correct'] + counts['correct_empty']
            accuracy = (correct / total * 100) if total > 0 else 0
            print(f"{col:<25} {correct:<8} {counts['missing']:<8} {counts['incorrect']:<8} {accuracy:>6.1f}%")
    
    def _print_missing_patterns(self):
        """Print the most common missing patterns"""
        print(f"\n🔍 TOP MISSING PATTERNS (should be added to training):")
        
        for col, patterns in self.missing_patterns.items():
            if patterns:
                print(f"\n   📊 {col}:")
                for (word, value), count in patterns.most_common(5):
                    print(f"      '{word}' → '{value}' (missing {count} times)")
    
    def _print_incorrect_predictions(self):
        """Print the most common incorrect predictions"""
        if not self.incorrect_predictions:
            print(f"\n✅ No incorrect predictions!")
            return
            
        print(f"\n❌ TOP INCORRECT PREDICTIONS (need pattern adjustment):")
        
        # Group by pattern
        incorrect_patterns = Counter()
        for error in self.incorrect_predictions:
            pattern = f"{error['column']}: '{error['predicted']}' instead of '{error['actual']}'"
            incorrect_patterns[pattern] += 1
        
        for pattern, count in incorrect_patterns.most_common(10):
            print(f"   {pattern} (wrong {count} times)")

def main():
    verifier = PredictionVerifier()
    
    # File paths
    predicted_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'PREDICTED_sample_data.xlsx')
    verified_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'verification', 'Filled_Sample_Data_For_Verification.xlsx')
    
    # Check if files exist
    if not os.path.exists(predicted_file):
        print(f"❌ Predicted file not found: {predicted_file}")
        return
        
    if not os.path.exists(verified_file):
        print(f"❌ Verified file not found: {verified_file}")
        return
    
    # Run verification
    accuracy = verifier.verify_predictions(predicted_file, verified_file)
    
    # Save improvement recommendations
    if accuracy < 90:
        print(f"\n💡 RECOMMENDATION: Accuracy is {accuracy:.1f}% - run pattern enhancement")
    else:
        print(f"\n🎉 EXCELLENT! Accuracy is {accuracy:.1f}% - system is working well!")

if __name__ == "__main__":
    main()