import pandas as pd
import os
from collections import defaultdict, Counter

class OriginalPlusVerifier:
    def __init__(self):
        self.stats = defaultdict(Counter)
    
    def verify_original_plus(self):
        """Verify original+ predictions"""
        predicted_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'output_original_plus', 'ORIGINAL_PLUS_sample_data.xlsx')
        verified_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'verification', 'Filled_Sample_Data_For_Verification.xlsx')
        
        print(f"🔍 Comparing ORIGINAL+ predictions...")
        print(f"   Predicted: {predicted_file}")
        print(f"   Verified:  {verified_file}")
        
        if not os.path.exists(predicted_file):
            print(f"❌ Original+ predicted file not found")
            return 0
            
        if not os.path.exists(verified_file):
            print(f"❌ Verified file not found")
            return 0
        
        pred_df = pd.read_excel(predicted_file)
        true_df = pd.read_excel(verified_file)
        
        # Get common columns
        common_columns = set(pred_df.columns) & set(true_df.columns)
        common_columns = [col for col in common_columns if col not in ['dataTagId', 'description']]
        
        merged = pd.merge(pred_df, true_df, on='dataTagId', suffixes=('_pred', '_true'))
        print(f"📊 Comparing {len(merged)} rows with {len(common_columns)} common columns")
        
        total_cells = 0
        correct_cells = 0
        
        for idx, row in merged.iterrows():
            for col in common_columns:
                pred_col = f"{col}_pred"
                true_col = f"{col}_true"
                
                if pred_col in row and true_col in row:
                    total_cells += 1
                    pred_val = row[pred_col]
                    true_val = row[true_col]
                    
                    if self._values_equal(pred_val, true_val):
                        correct_cells += 1
                        self.stats[col]['correct'] += 1
                    else:
                        if pd.isna(pred_val) or pred_val in ['', '-', 'Unassigned']:
                            self.stats[col]['missing'] += 1
                        else:
                            self.stats[col]['incorrect'] += 1
        
        accuracy = (correct_cells / total_cells * 100) if total_cells > 0 else 0
        print(f"\n🎯 ORIGINAL+ ACCURACY: {accuracy:.1f}% ({correct_cells}/{total_cells} cells)")
        
        self._print_detailed_report()
        
        if accuracy > 44.4:
            print(f"🎉 SUCCESS! Original+ is better than original 44.4%")
        elif accuracy == 44.4:
            print(f"✅ MATCHING: Original+ matches original 44.4%")
        else:
            print(f"🔻 REGRESSION: Original+ is worse than original 44.4%")
        
        return accuracy
    
    def _values_equal(self, pred_val, true_val):
        if pd.isna(pred_val) and pd.isna(true_val):
            return True
        elif pd.isna(pred_val) or pd.isna(true_val):
            return False
        return str(pred_val).strip() == str(true_val).strip()
    
    def _print_detailed_report(self):
        print(f"\n📈 ORIGINAL+ COLUMN ACCURACY:")
        print(f"{'Column':<25} {'Correct':<8} {'Missing':<8} {'Wrong':<8} {'Accuracy':<10}")
        print("-" * 65)
        
        for col, counts in self.stats.items():
            total = sum(counts.values())
            correct = counts['correct']
            accuracy = (correct / total * 100) if total > 0 else 0
            print(f"{col:<25} {correct:<8} {counts['missing']:<8} {counts['incorrect']:<8} {accuracy:>6.1f}%")

def main():
    verifier = OriginalPlusVerifier()
    accuracy = verifier.verify_original_plus()

if __name__ == "__main__":
    main()