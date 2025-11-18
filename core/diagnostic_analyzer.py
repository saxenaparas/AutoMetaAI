import pandas as pd
import json
import os
from collections import defaultdict

class DiagnosticAnalyzer:
    def __init__(self):
        self.issues = defaultdict(list)
    
    def compare_all_versions(self, original_file, original_plus_file, verified_file):
        """Compare Original vs Original+ vs Ground Truth"""
        print("🔍 COMPARING ALL VERSIONS: Original vs Original+ vs Ground Truth")
        
        # Check if files exist
        if not os.path.exists(original_file):
            print(f"❌ Original file not found: {original_file}")
            return
        if not os.path.exists(original_plus_file):
            print(f"❌ Original+ file not found: {original_plus_file}")
            return
        if not os.path.exists(verified_file):
            print(f"❌ Verified file not found: {verified_file}")
            return
        
        # Read all files
        orig_df = pd.read_excel(original_file)
        orig_plus_df = pd.read_excel(original_plus_file)
        true_df = pd.read_excel(verified_file)
        
        # Get common columns across all files
        common_columns = set(orig_df.columns) & set(orig_plus_df.columns) & set(true_df.columns)
        common_columns = [col for col in common_columns if col not in ['dataTagId', 'description']]
        
        print(f"📊 Comparing {len(common_columns)} common columns across {len(orig_df)} rows")
        print(f"📁 Files:")
        print(f"   • Original: {original_file}")
        print(f"   • Original+: {original_plus_file}")
        print(f"   • Ground Truth: {verified_file}")
        
        # Merge all files
        orig_merged = pd.merge(orig_df, true_df, on='dataTagId', suffixes=('_pred', '_true'))
        orig_plus_merged = pd.merge(orig_plus_df, true_df, on='dataTagId', suffixes=('_pred', '_true'))
        
        print(f"\n🎯 ACCURACY COMPARISON:")
        print(f"{'Column':<25} {'Original':<10} {'Original+':<10} {'Difference':<12}")
        print("-" * 60)
        
        for col in common_columns:
            orig_correct = 0
            orig_plus_correct = 0
            total = 0
            
            for idx in range(len(orig_merged)):
                # Safely get values
                try:
                    orig_val = orig_merged.iloc[idx][f'{col}_pred']
                    orig_plus_val = orig_plus_merged.iloc[idx][f'{col}_pred']
                    true_val = orig_merged.iloc[idx][f'{col}_true']
                except KeyError:
                    continue
                
                if pd.notna(true_val) and true_val not in ['', '-', 'Unassigned']:
                    total += 1
                    
                    # Check original
                    if self._values_equal(orig_val, true_val):
                        orig_correct += 1
                    
                    # Check original+
                    if self._values_equal(orig_plus_val, true_val):
                        orig_plus_correct += 1
                    else:
                        # Record what changed from original to original+
                        if pd.notna(orig_val) and self._values_equal(orig_val, true_val) and not self._values_equal(orig_plus_val, true_val):
                            self.issues[col].append(f"Was correct '{orig_val}' → now wrong '{orig_plus_val}'")
            
            if total > 0:
                orig_acc = (orig_correct / total * 100)
                orig_plus_acc = (orig_plus_correct / total * 100)
                difference = orig_plus_acc - orig_acc
                
                if difference > 5:
                    print(f"✅ {col:<23} {orig_acc:>6.1f}%    {orig_plus_acc:>6.1f}%    +{difference:>5.1f}%")
                elif difference < -5:
                    print(f"❌ {col:<23} {orig_acc:>6.1f}%    {orig_plus_acc:>6.1f}%    {difference:>6.1f}%")
                else:
                    print(f"➖ {col:<23} {orig_acc:>6.1f}%    {orig_plus_acc:>6.1f}%    {difference:>6.1f}%")
    
    def analyze_improvements_and_regressions(self, original_plus_file, verified_file):
        """Show specific improvements and regressions in Original+"""
        print(f"\n🔍 DETAILED ANALYSIS: What improved/regressed in Original+")
        
        if not os.path.exists(original_plus_file) or not os.path.exists(verified_file):
            print("❌ Files not found for detailed analysis")
            return
            
        orig_plus_df = pd.read_excel(original_plus_file)
        true_df = pd.read_excel(verified_file)
        
        merged = pd.merge(orig_plus_df, true_df, on='dataTagId', suffixes=('_pred', '_true'))
        
        # Show improvements (where Original+ fixed previous issues)
        print(f"\n📈 POTENTIAL IMPROVEMENTS in Original+:")
        improvement_found = False
        
        for col in ['measureLocation', 'measureLocationName', 'measureProperty', 'measureType']:
            print(f"\n   📊 {col}:")
            correct_count = 0
            total = 0
            
            for idx in range(len(merged)):
                try:
                    pred_val = merged.iloc[idx][f'{col}_pred']
                    true_val = merged.iloc[idx][f'{col}_true']
                except KeyError:
                    continue
                
                if pd.notna(true_val) and true_val not in ['', '-', 'Unassigned']:
                    total += 1
                    if self._values_equal(pred_val, true_val):
                        correct_count += 1
                        # Show some correct predictions that might be new
                        if correct_count <= 2 and pd.notna(pred_val):
                            desc = merged.iloc[idx].get('description_pred', '')
                            if 'I/L' in desc or 'O/L' in desc or 'SUCTION' in desc or 'DISCH' in desc:
                                print(f"      ✅ Correct: '{pred_val}' for '{desc}'")
                                improvement_found = True
            
            if total > 0:
                accuracy = (correct_count / total * 100)
                print(f"      Accuracy: {accuracy:.1f}%")
        
        if not improvement_found:
            print("      No clear improvements detected")
        
        # Show regressions
        print(f"\n📉 POTENTIAL REGRESSIONS in Original+:")
        regression_found = False
        
        for col, issues in self.issues.items():
            if issues:
                print(f"\n   ❌ {col}:")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"      {issue}")
                    regression_found = True
        
        if not regression_found:
            print("      No major regressions detected")
    
    def check_original_patterns(self):
        """Check the original pattern file"""
        print(f"\n📁 ORIGINAL PATTERN ANALYSIS:")
        
        pattern_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'pattern_knowledge.json')
        
        if not os.path.exists(pattern_file):
            print(f"❌ Pattern file not found: {pattern_file}")
            return
        
        with open(pattern_file, 'r') as f:
            patterns = json.load(f)
        
        # Check key patterns we care about
        key_columns = ['measureLocation', 'measureLocationName', 'measureType', 'measureProperty']
        
        for col in key_columns:
            print(f"\n📊 {col} patterns:")
            if col in patterns:
                count = len(patterns[col])
                print(f"   Total patterns: {count}")
                
                # Show some relevant patterns
                relevant_patterns = []
                for word, data in patterns[col].items():
                    if any(keyword in word for keyword in ['I/L', 'O/L', 'SUCTION', 'DISCH', 'TEMP', 'PRESS']):
                        relevant_patterns.append((word, data))
                
                for word, data in relevant_patterns[:5]:  # Show top 5
                    print(f"   '{word}' → '{data['value']}' (conf: {data['confidence']})")
            else:
                print(f"   No patterns found for this column")
    
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
    original_plus_file = os.path.join(base_dir, '..', 'data', 'output_original_plus', 'ORIGINAL_PLUS_sample_data.xlsx')
    verified_file = os.path.join(base_dir, '..', 'data', 'verification', 'Filled_Sample_Data_For_Verification.xlsx')
    
    print("🎯 RUNNING ORIGINAL vs ORIGINAL+ DIAGNOSTIC...")
    
    # Step 1: Compare all versions
    diagnostic.compare_all_versions(original_file, original_plus_file, verified_file)
    
    # Step 2: Analyze improvements and regressions
    diagnostic.analyze_improvements_and_regressions(original_plus_file, verified_file)
    
    # Step 3: Check original patterns
    diagnostic.check_original_patterns()
    
    print(f"\n💡 FINAL ASSESSMENT: See if Original+ maintained or improved on 44.4% accuracy")

if __name__ == "__main__":
    main()