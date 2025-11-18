import pandas as pd
import json
import os
import glob
from collections import defaultdict, Counter

class MetaDataPredictor:
    def __init__(self, pattern_file=None):
        if pattern_file is None:
            pattern_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'pattern_knowledge.json')
        
        self.pattern_file = os.path.abspath(pattern_file)
        print(f"📂 Loading patterns from: {self.pattern_file}")
        
        if not os.path.exists(self.pattern_file):
            print(f"❌ Pattern file not found: {self.pattern_file}")
            self.patterns = {}
            return
            
        with open(self.pattern_file, 'r') as f:
            self.patterns = json.load(f)
        
        print(f"✅ Loaded {sum(len(v) for v in self.patterns.values())} patterns across {len(self.patterns)} columns")
    
    def predict_all_files(self, input_dir=None, output_dir=None):
        """Predict all Excel files in input directory"""
        if input_dir is None:
            input_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'input')
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
        
        # Create directories if they don't exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 Input directory: {input_dir}")
        print(f"📁 Output directory: {output_dir}")
        
        # Find all Excel files in input directory
        excel_files = glob.glob(os.path.join(input_dir, "*.xlsx")) + glob.glob(os.path.join(input_dir, "*.xls"))
        
        if not excel_files:
            print("❌ No Excel files found in input directory!")
            print("💡 Place your Excel files (with only 'dataTagId' and 'description' columns) in:")
            print(f"   {input_dir}")
            return
        
        print(f"📊 Found {len(excel_files)} Excel file(s) to process")
        
        for input_file in excel_files:
            filename = os.path.basename(input_file)
            output_file = os.path.join(output_dir, f"PREDICTED_{filename}")
            print(f"\n🎯 Processing: {filename}")
            self.predict_single_file(input_file, output_file)
    
    def predict_single_file(self, input_file, output_file):
        """Predict a single Excel file"""
        print(f"📥 Reading: {input_file}")
        
        try:
            df = pd.read_excel(input_file)
            print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")
            
            # Validate required columns
            if 'dataTagId' not in df.columns or 'description' not in df.columns:
                print("❌ ERROR: File must contain 'dataTagId' and 'description' columns")
                return
            
            # Add empty columns if they don't exist
            expected_columns = ['system', 'systemName', 'systemInstance', 'equipment', 
                              'equipmentName', 'equipmentInstance', 'component', 'componentName', 
                              'componentInstance', 'subcomponent', 'subcomponentName', 
                              'subcomponentInstance', 'measureLocation', 'measureLocationName', 
                              'measureLocationInstance', 'measureProperty', 'measureType', 
                              'measureUnit', 'tagType', 'address']
            
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Predict each row
            predictions = []
            for idx, row in df.iterrows():
                predicted_row = self._predict_single_row(row)
                predictions.append(predicted_row)
                
                if (idx + 1) % 100 == 0:
                    print(f"   📝 Processed {idx+1} rows...")
            
            # Create result dataframe
            result_df = pd.DataFrame(predictions)
            result_df.to_excel(output_file, index=False)
            print(f"✅ Saved: {output_file}")
            
            # Show prediction statistics
            self._show_prediction_stats(result_df)
            
        except Exception as e:
            print(f"❌ Error processing {input_file}: {e}")
    
    def _predict_single_row(self, row):
        """Predict columns for a single row"""
        result = row.to_dict()
        description = str(row['description']).upper() if pd.notna(row['description']) else ""
        
        if not description:
            return result
            
        words = description.split()
        
        # For each column, use pattern matching
        for col, patterns in self.patterns.items():
            if col not in result or pd.isna(result[col]) or result[col] in ['', '-', 'Unassigned']:
                column_predictions = Counter()
                
                for word in words:
                    if word in patterns:
                        pattern_data = patterns[word]
                        # Weight by confidence and occurrences
                        score = pattern_data['confidence'] * pattern_data['occurrences']
                        column_predictions[pattern_data['value']] += score
                
                if column_predictions:
                    best_prediction, best_score = column_predictions.most_common(1)[0]
                    # Only use prediction if we have reasonable confidence
                    if best_score > 10:
                        result[col] = best_prediction
        
        return result
    
    def _show_prediction_stats(self, result_df):
        """Show statistics about predictions"""
        total_cells = len(result_df) * len(self.patterns)
        filled_cells = 0
        
        print("   📊 Prediction Summary:")
        for col in self.patterns.keys():
            if col in result_df.columns:
                non_empty = result_df[col].notna() & ~result_df[col].isin(['', '-', 'Unassigned'])
                filled_count = non_empty.sum()
                filled_cells += filled_count
                if filled_count > 0:
                    print(f"      {col}: {filled_count}/{len(result_df)} rows filled")
        
        fill_percentage = (filled_cells / total_cells * 100) if total_cells > 0 else 0
        print(f"   🎯 Overall: {filled_cells}/{total_cells} cells filled ({fill_percentage:.1f}%)")

def main():
    predictor = MetaDataPredictor()
    
    # Process ALL files in input directory
    predictor.predict_all_files()

if __name__ == "__main__":
    main()