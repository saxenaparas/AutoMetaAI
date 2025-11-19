import pandas as pd
import json
import os
import glob
from collections import defaultdict, Counter

class RuleBasedPredictor:
    def __init__(self, rules_file=None):
        if rules_file is None:
            rules_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'pattern_rules.json')
        
        self.rules_file = os.path.abspath(rules_file)
        print(f"📂 Loading rules from: {self.rules_file}")
        
        if not os.path.exists(self.rules_file):
            print(f"❌ Rules file not found: {self.rules_file}")
            self.rules = {}
            return
            
        with open(self.rules_file, 'r') as f:
            self.rules = json.load(f)
        
        print(f"✅ Loaded rules:")
        print(f"   - Column Relationships: {len(self.rules.get('column_relationships', {}))}")
        print(f"   - Word Mappings: {sum(len(v) for v in self.rules.get('word_mappings', {}).values())}")
        print(f"   - Conditional Rules: {len(self.rules.get('conditional_rules', []))}")
        print(f"   - Hierarchical Patterns: {len(self.rules.get('hierarchical_patterns', {}))}")
    
    def predict_single_file(self, input_file, output_file):
        """Predict a single Excel file using discovered rules"""
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
            
            # Predict each row using rules
            predictions = []
            for idx, row in df.iterrows():
                predicted_row = self._apply_rules(row)
                predictions.append(predicted_row)
                
                if (idx + 1) % 10 == 0:
                    print(f"   📝 Processed {idx+1} rows...")
            
            # Create result dataframe
            result_df = pd.DataFrame(predictions)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            
            result_df.to_excel(output_file, index=False)
            print(f"✅ Saved: {output_file}")
            
            # Show prediction statistics
            self._show_prediction_stats(result_df)
            
            return result_df
            
        except Exception as e:
            print(f"❌ Error processing {input_file}: {e}")
            return None
    
    def _apply_rules(self, row):
        """Apply all discovered rules to a single row"""
        result = row.to_dict()
        description = str(row['description']).upper() if pd.notna(row['description']) else ""
        
        if not description:
            return result
        
        words = description.split()
        
        # Step 1: Apply word mappings (basic patterns)
        self._apply_word_mappings(result, words)
        
        # Step 2: Apply column relationships (* and + notation)
        self._apply_column_relationships(result)
        
        # Step 3: Apply conditional rules (if-then logic)
        self._apply_conditional_rules(result, description, words)
        
        # Step 4: Apply hierarchical patterns
        self._apply_hierarchical_patterns(result)
        
        return result
    
    def _apply_word_mappings(self, result, words):
        """Apply word -> value mappings"""
        word_mappings = self.rules.get('word_mappings', {})
        
        for col, mappings in word_mappings.items():
            if self._is_empty(result.get(col)):
                column_predictions = Counter()
                
                for word in words:
                    if word in mappings:
                        mapping_data = mappings[word]
                        score = mapping_data['confidence'] * mapping_data['occurrences']
                        column_predictions[mapping_data['value']] += score
                
                if column_predictions:
                    best_prediction, best_score = column_predictions.most_common(1)[0]
                    if best_score > 10:  # Confidence threshold
                        result[col] = best_prediction
    
    def _apply_column_relationships(self, result):
        """Apply * and + relationships between columns"""
        relationships = self.rules.get('column_relationships', {})
        
        for target_col, relationship in relationships.items():
            if self._is_empty(result.get(target_col)):
                rel_type = relationship.get('type')
                source_col = relationship.get('source')
                
                if rel_type == 'equals' and source_col in result and not self._is_empty(result[source_col]):
                    # systemName = system *
                    result[target_col] = result[source_col]
                elif rel_type == 'default_value':
                    # instance columns = "1" or "1.0"
                    result[target_col] = relationship.get('value')
    
    def _apply_conditional_rules(self, result, description, words):
        """Apply if-then conditional rules"""
        conditional_rules = self.rules.get('conditional_rules', [])
        
        for rule in conditional_rules:
            condition = rule.get('condition', {})
            then_action = rule.get('then', {})
            
            # Check if condition is met
            if self._check_condition(description, words, condition):
                # Apply the "then" action
                for col, value in then_action.items():
                    if self._is_empty(result.get(col)):
                        result[col] = value
    
    def _apply_hierarchical_patterns(self, result):
        """Apply equipment -> system hierarchies"""
        hierarchical_patterns = self.rules.get('hierarchical_patterns', {})
        
        equipment = result.get('equipment')
        if equipment and equipment in hierarchical_patterns and self._is_empty(result.get('system')):
            system = hierarchical_patterns[equipment].get('system')
            if system:
                result['system'] = system
                result['systemName'] = system  # systemName = system *
    
    def _check_condition(self, description, words, condition):
        """Check if a condition is met"""
        contains_words = condition.get('description_contains', [])
        
        if contains_words:
            for word in contains_words:
                if word in description:
                    return True
        return False
    
    def _is_empty(self, value):
        """Check if a value is empty or unassigned"""
        return (pd.isna(value) or value in ['', '-', 'Unassigned', None])
    
    def _show_prediction_stats(self, result_df):
        """Show statistics about predictions"""
        expected_columns = ['system', 'systemName', 'systemInstance', 'equipment', 
                          'equipmentName', 'equipmentInstance', 'component', 'componentName', 
                          'componentInstance', 'subcomponent', 'subcomponentName', 
                          'subcomponentInstance', 'measureLocation', 'measureLocationName', 
                          'measureLocationInstance', 'measureProperty', 'measureType', 
                          'measureUnit']
        
        total_cells = len(result_df) * len(expected_columns)
        filled_cells = 0
        
        print("   📊 Prediction Summary:")
        for col in expected_columns:
            if col in result_df.columns:
                non_empty = result_df[col].notna() & ~result_df[col].isin(['', '-', 'Unassigned'])
                filled_count = non_empty.sum()
                filled_cells += filled_count
                if filled_count > 0:
                    print(f"      {col}: {filled_count}/{len(result_df)} rows filled")
        
        fill_percentage = (filled_cells / total_cells * 100) if total_cells > 0 else 0
        print(f"   🎯 Overall: {filled_cells}/{total_cells} cells filled ({fill_percentage:.1f}%)")

def main():
    predictor = RuleBasedPredictor()
    
    # Test prediction
    input_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'input', 'sample_data.xlsx')
    output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'RULES_PREDICTED_sample_data.xlsx')
    
    if os.path.exists(input_file):
        predictor.predict_single_file(input_file, output_file)
    else:
        print(f"❌ Input file not found: {input_file}")

if __name__ == "__main__":
    main()