import pandas as pd
import os
from collections import defaultdict, Counter
import json

class PatternAnalyzer:
    def __init__(self):
        self.patterns = defaultdict(Counter)
        self.column_stats = defaultdict(Counter)
        
    def analyze_files(self, training_dir):
        """Analyze all Excel files in training directory"""
        # Convert to absolute path
        training_dir = os.path.abspath(training_dir)
        print(f"📂 Looking in: {training_dir}")
        
        if not os.path.exists(training_dir):
            print(f"❌ ERROR: Directory doesn't exist: {training_dir}")
            return {}
            
        excel_files = [f for f in os.listdir(training_dir) if f.endswith(('.xlsx', '.xls'))]
        print(f"📊 Found {len(excel_files)} Excel files")
        
        total_rows = 0
        for file in excel_files:
            file_path = os.path.join(training_dir, file)
            print(f"   Processing: {file}")
            
            try:
                df = pd.read_excel(file_path)
                print(f"      Rows: {len(df)}, Columns: {len(df.columns)}")
                total_rows += len(df)
                self._analyze_dataframe(df)
            except Exception as e:
                print(f"      ❌ Error reading {file}: {e}")
            
        print(f"✅ Analyzed {total_rows} total rows")
        return self._generate_pattern_report()
    
    def _analyze_dataframe(self, df):
        """Analyze patterns in a single dataframe"""
        for _, row in df.iterrows():
            description = str(row.iloc[1]).upper() if len(row) > 1 else ""  # 2nd column
            words = description.split()
            
            # Analyze each column (skip first two: tag_id and description)
            for col_idx, col_name in enumerate(df.columns[2:], 2):
                value = row.iloc[col_idx] if col_idx < len(row) else None
                if pd.notna(value) and value not in ['-', '', 'Unassigned', None]:
                    for word in words:
                        self.patterns[(col_name, word)][str(value)] += 1
                    self.column_stats[col_name][str(value)] += 1
    
    def _generate_pattern_report(self):
        """Generate pattern confidence scores"""
        pattern_report = {}
        
        for (col, word), value_counts in self.patterns.items():
            total = sum(value_counts.values())
            if total >= 2:  # Only consider patterns with at least 2 occurrences
                most_common_value, count = value_counts.most_common(1)[0]
                confidence = count / total
                
                if confidence >= 0.7:  # 70% confidence threshold
                    if col not in pattern_report:
                        pattern_report[col] = {}
                    pattern_report[col][word] = {
                        'value': most_common_value,
                        'confidence': round(confidence, 3),
                        'occurrences': count
                    }
        
        print(f"🎯 Discovered {sum(len(v) for v in pattern_report.values())} high-confidence patterns")
        return pattern_report

def main():
    analyzer = PatternAnalyzer()
    
    # Use absolute path - adjust this to your actual data location
    training_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'training')
    
    patterns = analyzer.analyze_files(training_dir)
    
    if patterns:
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save patterns to file
        pattern_file = os.path.join(output_dir, 'pattern_knowledge.json')
        with open(pattern_file, 'w') as f:
            json.dump(patterns, f, indent=2)
        
        print(f"💾 Patterns saved to: {pattern_file}")
        
        # Print top patterns
        for col, word_patterns in patterns.items():
            print(f"\n📈 {col}:")
            for word, data in list(word_patterns.items())[:5]:
                print(f"   '{word}' → '{data['value']}' (conf: {data['confidence']}, occ: {data['occurrences']})")
    else:
        print("❌ No patterns found. Check your file paths and data.")

if __name__ == "__main__":
    main()