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
        
        print(f"✅ Loaded enhanced rules:")
        print(f"   - Column Relationships: {len(self.rules.get('column_relationships', {}))}")
        print(f"   - Word Mappings: {sum(len(v) for v in self.rules.get('word_mappings', {}).values())}")
        print(f"   - Conditional Rules: {len(self.rules.get('conditional_rules', []))}")
        print(f"   - Hierarchical Patterns: {len(self.rules.get('hierarchical_patterns', {}))}")
        print(f"   - Negative Patterns: {len(self.rules.get('negative_patterns', {}))}")
    
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
            
            # Predict each row using enhanced rules
            predictions = []
            for idx, row in df.iterrows():
                predicted_row = self._apply_enhanced_rules(row)
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
    
    def _apply_enhanced_rules(self, row):
        """Apply all discovered rules to a single row with enhanced logic"""
        result = row.to_dict()
        description = str(row['description']).upper() if pd.notna(row['description']) else ""
        
        if not description:
            return result
        
        words = description.split()
        
        # Step 0: Check negative patterns first (what NOT to predict)
        self._apply_negative_patterns(result, description, words)
        
        # Step 1: Apply conditional rules (highest priority)
        self._apply_conditional_rules(result, description, words)
        
        # Step 2: Apply word mappings (basic patterns)
        self._apply_word_mappings(result, words)
        
        # Step 3: Apply column relationships (* and + notation)
        self._apply_column_relationships(result)
        
        # Step 4: Apply hierarchical patterns
        self._apply_hierarchical_patterns(result)
        
        # Step 5: Apply measurement relationships
        self._apply_measurement_relationships(result)
        
        return result
    
    def _apply_negative_patterns(self, result, description, words):
        """Apply negative patterns to avoid common errors"""
        negative_patterns = self.rules.get('negative_patterns', {})
        
        for col, pattern in negative_patterns.items():
            conditions = pattern.get('conditions', [])
            action = pattern.get('action')
            
            # Check if negative conditions are met
            if self._check_negative_conditions(description, words, conditions):
                if action == 'skip_prediction' and col in result:
                    # Don't predict this column if conditions are met
                    result[col] = None
    
    def _apply_conditional_rules(self, result, description, words):
        """Apply if-then conditional rules with priority"""
        conditional_rules = self.rules.get('conditional_rules', [])
        
        # Sort by priority (highest first)
        conditional_rules.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        for rule in conditional_rules:
            condition = rule.get('condition', {})
            then_action = rule.get('then', {})
            
            # Check if condition is met
            if self._check_condition(description, words, condition):
                # Apply the "then" action
                for col, value in then_action.items():
                    if self._should_predict(result.get(col)):
                        result[col] = value
    
    def _apply_word_mappings(self, result, words):
        """Apply word -> value mappings with confidence thresholds"""
        word_mappings = self.rules.get('word_mappings', {})
        
        for col, mappings in word_mappings.items():
            if self._should_predict(result.get(col)):
                column_predictions = Counter()
                
                for word in words:
                    if word in mappings:
                        mapping_data = mappings[word]
                        # Use confidence and occurrences for scoring
                        score = mapping_data['confidence'] * mapping_data['occurrences']
                        column_predictions[mapping_data['value']] += score
                
                if column_predictions:
                    best_prediction, best_score = column_predictions.most_common(1)[0]
                    # Use dynamic threshold based on confidence
                    if best_score > 15:  # Increased threshold for better accuracy
                        result[col] = best_prediction
    
    def _apply_column_relationships(self, result):
        """Apply * and + relationships between columns"""
        relationships = self.rules.get('column_relationships', {})
        
        for target_col, relationship in relationships.items():
            if self._should_predict(result.get(target_col)):
                rel_type = relationship.get('type')
                
                if rel_type == 'equals':
                    source_col = relationship.get('source')
                    if source_col in result and not self._is_empty(result[source_col]):
                        # systemName = system *
                        result[target_col] = result[source_col]
                elif rel_type == 'default_value':
                    # instance columns = "1" or "1.0"
                    result[target_col] = relationship.get('value')
                elif rel_type == 'plus':
                    source_col = relationship.get('source')
                    if source_col in result and not self._is_empty(result[source_col]):
                        # equipmentName = equipment +
                        base_value = result[source_col]
                        # For now, use the base value (in practice, you'd extract modifiers from description)
                        result[target_col] = base_value
    
    def _apply_hierarchical_patterns(self, result):
        """Apply equipment -> system hierarchies"""
        hierarchical_patterns = self.rules.get('hierarchical_patterns', {})
        
        equipment = result.get('equipment')
        if equipment and equipment in hierarchical_patterns and self._should_predict(result.get('system')):
            system = hierarchical_patterns[equipment].get('system')
            if system:
                result['system'] = system
                if self._should_predict(result.get('systemName')):
                    result['systemName'] = system  # systemName = system *
    
    def _apply_measurement_relationships(self, result):
        """Apply relationships between measurement types and units"""
        relationships = self.rules.get('column_relationships', {})
        
        # Temperature -> Degc
        if (self._safe_compare(result.get('measureType'), 'Temperature') and 
            self._should_predict(result.get('measureUnit'))):
            result['measureUnit'] = 'Degc'
        
        # Pressure -> Kg/Cm2  
        if (self._safe_compare(result.get('measureType'), 'Pressure') and 
            self._should_predict(result.get('measureUnit'))):
            result['measureUnit'] = 'Kg/Cm2'
    
    def _check_negative_conditions(self, description, words, conditions):
        """Check negative conditions (what should NOT trigger predictions)"""
        for condition in conditions:
            contains_words = condition.get('description_contains', [])
            not_contains_words = condition.get('description_not_contains', [])
            
            # Check if it contains unwanted words
            contains_match = any(word in description for word in contains_words) if contains_words else True
            # Check if it doesn't contain required words
            not_contains_match = not any(word in description for word in not_contains_words) if not_contains_words else True
            
            if contains_match and not_contains_match:
                return True
        return False
    
    def _check_condition(self, description, words, condition):
        """Check if a condition is met"""
        contains_words = condition.get('description_contains', [])
        
        if contains_words:
            for word in contains_words:
                if word in description:
                    return True
        return False
    
    def _should_predict(self, value):
        """Check if we should predict a value (it's empty or unassigned)"""
        return self._is_empty(value)
    
    def _is_empty(self, value):
        """Check if a value is empty or unassigned"""
        return (pd.isna(value) or value in ['', '-', 'Unassigned', None])
    
    def _safe_compare(self, val1, val2):
        """Safely compare two values that might be different types"""
        if pd.isna(val1) or val1 is None:
            return False
        return str(val1).strip().upper() == str(val2).strip().upper()
    
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
    output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'ENHANCED_PREDICTED_sample_data.xlsx')
    
    if os.path.exists(input_file):
        predictor.predict_single_file(input_file, output_file)
    else:
        print(f"❌ Input file not found: {input_file}")

if __name__ == "__main__":
    main()