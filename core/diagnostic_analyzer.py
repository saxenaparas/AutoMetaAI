import pandas as pd
import json
import os
from collections import defaultdict

class DiagnosticAnalyzer:
    def __init__(self):
        self.issues = defaultdict(list)
    
    def compare_predictions(self, original_file, balanced_file, verified_file):
        """Compare original vs balanced predictions to find what broke"""
        print("🔍 DIAGNOSTIC ANALYSIS: What broke between original and balanced?")
        
        # Check if files exist
        if not os.path.exists(original_file):
            print(f"❌ Original file not found: {original_file}")
            return
        if not os.path.exists(balanced_file):
            print(f"❌ Balanced file not found: {balanced_file}")
            return
        if not os.path.exists(verified_file):
            print(f"❌ Verified file not found: {verified_file}")
            return
        
        # Read all files
        orig_df = pd.read_excel(original_file)
        bal_df = pd.read_excel(balanced_file)
        true_df = pd.read_excel(verified_file)
        
        # Get common columns across all files
        common_columns = set(orig_df.columns) & set(bal_df.columns) & set(true_df.columns)
        common_columns = [col for col in common_columns if col not in ['dataTagId', 'description']]
        
        print(f"📊 Comparing {len(common_columns)} common columns across {len(orig_df)} rows")
        
        # Merge on dataTagId
        orig_merged = pd.merge(orig_df, true_df, on='dataTagId', suffixes=('_pred', '_true'))
        bal_merged = pd.merge(bal_df, true_df, on='dataTagId', suffixes=('_pred', '_true'))
        
        for col in common_columns:
            orig_correct = 0
            bal_correct = 0
            total = 0
            
            for idx in range(len(orig_merged)):
                # Safely get values with error handling
                try:
                    orig_val = orig_merged.iloc[idx][f'{col}_pred']
                    bal_val = bal_merged.iloc[idx][f'{col}_pred']
                    true_val = orig_merged.iloc[idx][f'{col}_true']
                except KeyError:
                    # Skip if column doesn't exist in merged data
                    continue
                
                if pd.notna(true_val) and true_val not in ['', '-', 'Unassigned']:
                    total += 1
                    
                    # Check original
                    if self._values_equal(orig_val, true_val):
                        orig_correct += 1
                    
                    # Check balanced
                    if self._values_equal(bal_val, true_val):
                        bal_correct += 1
                    else:
                        # Record what went wrong
                        if pd.notna(orig_val) and self._values_equal(orig_val, true_val):
                            self.issues[col].append(f"Changed from correct '{orig_val}' to wrong '{bal_val}'")
            
            if total > 0:  # Only show columns with data
                orig_acc = (orig_correct / total * 100)
                bal_acc = (bal_correct / total * 100)
                
                if bal_acc < orig_acc - 5:  # Significant regression
                    print(f"❌ {col}: {orig_acc:.1f}% → {bal_acc:.1f}% (REGression: -{orig_acc-bal_acc:.1f}%)")
                elif bal_acc > orig_acc + 5:  # Significant improvement
                    print(f"✅ {col}: {orig_acc:.1f}% → {bal_acc:.1f}% (IMPROVEMENT: +{bal_acc-orig_acc:.1f}%)")
                else:
                    print(f"➖ {col}: {orig_acc:.1f}% → {bal_acc:.1f}%")
    
    def analyze_instance_columns(self, balanced_file, verified_file):
        """Specifically analyze why instance columns broke"""
        print(f"\n🔧 ANALYZING INSTANCE COLUMNS:")
        
        if not os.path.exists(balanced_file) or not os.path.exists(verified_file):
            print("❌ Files not found for instance analysis")
            return
            
        bal_df = pd.read_excel(balanced_file)
        true_df = pd.read_excel(verified_file)
        
        # Get common columns
        common_columns = set(bal_df.columns) & set(true_df.columns)
        instance_cols = ['systemInstance', 'equipmentInstance', 'componentInstance', 'subcomponentInstance']
        instance_cols = [col for col in instance_cols if col in common_columns]
        
        merged = pd.merge(bal_df, true_df, on='dataTagId', suffixes=('_pred', '_true'))
        
        for col in instance_cols:
            print(f"\n📊 {col}:")
            wrong_count = 0
            total = 0
            
            for idx in range(len(merged)):
                try:
                    pred_val = merged.iloc[idx][f'{col}_pred']
                    true_val = merged.iloc[idx][f'{col}_true']
                except KeyError:
                    continue
                
                if pd.notna(true_val) and true_val not in ['', '-', 'Unassigned']:
                    total += 1
                    if not self._values_equal(pred_val, true_val):
                        wrong_count += 1
                        if wrong_count <= 3:  # Show only first 3 errors
                            desc = merged.iloc[idx].get('description_pred', '')
                            print(f"   ❌ Row {idx}: predicted '{pred_val}' but should be '{true_val}'")
                            if desc:
                                print(f"      Description: {desc}")
            
            if total > 0:
                accuracy = ((total - wrong_count) / total * 100)
                print(f"   Accuracy: {accuracy:.1f}% ({total - wrong_count}/{total} correct)")
            else:
                print(f"   No data to compare")
    
    def check_pattern_files(self):
        """Check what changed in pattern files"""
        print(f"\n📁 PATTERN FILE COMPARISON:")
        
        original_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'pattern_knowledge.json')
        balanced_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'pattern_knowledge_balanced.json')
        
        if not os.path.exists(original_file):
            print(f"❌ Original pattern file not found: {original_file}")
            return
        if not os.path.exists(balanced_file):
            print(f"❌ Balanced pattern file not found: {balanced_file}")
            return
        
        with open(original_file, 'r') as f:
            orig_patterns = json.load(f)
        
        with open(balanced_file, 'r') as f:
            bal_patterns = json.load(f)
        
        # Check instance columns
        instance_cols = ['systemInstance', 'equipmentInstance', 'componentInstance', 'subcomponentInstance']
        
        for col in instance_cols:
            print(f"\n📊 {col} patterns:")
            
            if col in orig_patterns and col in bal_patterns:
                orig_count = len(orig_patterns[col])
                bal_count = len(bal_patterns[col])
                print(f"   Original: {orig_count} patterns")
                print(f"   Balanced: {bal_count} patterns")
                
                # Check if patterns changed
                different_patterns = 0
                for word in orig_patterns[col]:
                    if word in bal_patterns[col]:
                        if orig_patterns[col][word]['value'] != bal_patterns[col][word]['value']:
                            different_patterns += 1
                            print(f"   🔄 '{word}': '{orig_patterns[col][word]['value']}' → '{bal_patterns[col][word]['value']}'")
                
                if different_patterns == 0:
                    print(f"   ✅ No pattern value changes")
            else:
                if col not in orig_patterns:
                    print(f"   ❌ Column not in original patterns")
                if col not in bal_patterns:
                    print(f"   ❌ Column not in balanced patterns")
    
    def _values_equal(self, val1, val2):
        """Flexible value comparison"""
        if pd.isna(val1) and pd.isna(val2):
            return True
        elif pd.isna(val1) or pd.isna(val2):
            return False
        
        # Convert to string and normalize
        str1 = str(val1).strip().lower()
        str2 = str(val2).strip().lower()
        
        # Handle '1' vs '1.0' etc.
        if str1.endswith('.0'):
            str1 = str1[:-2]
        if str2.endswith('.0'):
            str2 = str2[:-2]
        
        return str1 == str2

def main():
    diagnostic = DiagnosticAnalyzer()
    
    # Use ABSOLUTE paths
    base_dir = os.path.dirname(__file__)
    original_file = os.path.join(base_dir, '..', 'data', 'output', 'PREDICTED_sample_data.xlsx')
    balanced_file = os.path.join(base_dir, '..', 'data', 'output_balanced', 'BALANCED_sample_data.xlsx')
    verified_file = os.path.join(base_dir, '..', 'data', 'verification', 'Filled_Sample_Data_For_Verification.xlsx')
    
    print("🎯 RUNNING COMPREHENSIVE DIAGNOSTIC...")
    
    # Step 1: Compare predictions
    diagnostic.compare_predictions(original_file, balanced_file, verified_file)
    
    # Step 2: Analyze instance columns specifically
    diagnostic.analyze_instance_columns(balanced_file, verified_file)
    
    # Step 3: Check pattern files
    diagnostic.check_pattern_files()
    
    print(f"\n💡 RECOMMENDATION: Let's revert to original patterns and incrementally improve")

if __name__ == "__main__":
    main()