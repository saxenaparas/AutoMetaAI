import pandas as pd
import json
import os
from collections import defaultdict, Counter
import re

class PatternThinker:
    def __init__(self):
        self.rules = {
            'column_relationships': {},
            'word_mappings': {},
            'conditional_rules': [],
            'priority_rules': [],
            'hierarchical_patterns': {},
            'negative_patterns': {},  # NEW: Patterns to avoid
            'confidence_adjustments': {}  # NEW: Adjust confidence based on errors
        }
        self.iteration = 0
    
    def analyze_training_data(self, training_dir, feedback_data=None):
        """Deep analysis to discover complex patterns with feedback learning"""
        self.iteration += 1
        print(f"THINKING: Analyzing training data for patterns (Iteration {self.iteration})...")
        
        # Load all training files
        all_data = self._load_all_training_data(training_dir)
        
        # Apply feedback from previous errors
        if feedback_data:
            print("   Applying feedback from previous errors...")
            all_data = self._apply_feedback(all_data, feedback_data)
        
        # Discover different types of patterns
        self._discover_column_relationships(all_data)
        self._discover_word_mappings(all_data) 
        self._discover_conditional_rules(all_data)
        self._discover_priority_rules(all_data)
        self._discover_hierarchical_patterns(all_data)
        self._discover_negative_patterns(all_data)  # NEW: What NOT to predict
        
        return self.rules
    
    def _apply_feedback(self, data, feedback_data):
        """Apply feedback from verification errors to improve patterns"""
        # This is a simplified version - in practice, you'd want more sophisticated feedback application
        print(f"      Processing {len(feedback_data.get('errors', []))} feedback items")
        
        for error in feedback_data.get('errors', []):
            # Lower confidence for patterns that caused errors
            col = error['column']
            wrong_value = error['predicted']
            description = error['description']
            
            words = description.upper().split()
            for word in words:
                pattern_key = f"{col}::{word}::{wrong_value}"
                if pattern_key not in self.rules['confidence_adjustments']:
                    self.rules['confidence_adjustments'][pattern_key] = 0
                self.rules['confidence_adjustments'][pattern_key] -= 1  # Penalize wrong patterns
        
        return data
    
    def _discover_column_relationships(self, data):
        """Discover * and + relationships between columns"""
        print("   Analyzing column relationships...")
        
        # Clear previous relationships
        self.rules['column_relationships'] = {}
        
        # systemName = system *
        system_matches = sum(1 for row in data if self._safe_compare(row.get('system'), row.get('systemName')))
        system_total = sum(1 for row in data if row.get('system') and row.get('systemName'))
        if system_total > 0 and system_matches / system_total > 0.8:
            self.rules['column_relationships']['systemName'] = {
                'type': 'equals', 
                'source': 'system', 
                'confidence': system_matches/system_total,
                'notation': '*'
            }
        
        # equipmentName = equipment +
        equipment_plus = self._analyze_plus_relationship(data, 'equipment', 'equipmentName')
        if equipment_plus:
            self.rules['column_relationships']['equipmentName'] = equipment_plus
            
        # Add instance relationships (usually "1" or "1.0")
        self._analyze_instance_relationships(data)
        
        # NEW: Discover more column relationships
        self._discover_measurement_relationships(data)
    
    def _discover_word_mappings(self, data):
        """Discover word -> value mappings with confidence, applying feedback adjustments"""
        print("   Analyzing word mappings...")
        
        word_mappings = defaultdict(Counter)
        
        for row in data:
            description = self._safe_str(row.get('description', '')).upper()
            words = description.split()
            
            for col, value in row.items():
                if col not in ['dataTagId', 'description'] and pd.notna(value) and value not in ['', '-']:
                    safe_value = self._safe_str(value)
                    for word in words:
                        word_mappings[(col, word)][safe_value] += 1
        
        # Convert to confidence scores with feedback adjustments
        for (col, word), value_counts in word_mappings.items():
            total = sum(value_counts.values())
            if total >= 2:  # Only consider patterns with at least 2 occurrences
                for value, count in value_counts.items():
                    confidence = count / total
                    
                    # Apply feedback adjustments
                    pattern_key = f"{col}::{word}::{value}"
                    if pattern_key in self.rules['confidence_adjustments']:
                        adjustment = self.rules['confidence_adjustments'][pattern_key]
                        confidence = max(0.1, confidence + (adjustment * 0.1))  # Adjust confidence
                    
                    if confidence >= 0.7:  # 70% confidence threshold
                        if col not in self.rules['word_mappings']:
                            self.rules['word_mappings'][col] = {}
                        
                        # Only keep the highest confidence mapping for each word
                        if word not in self.rules['word_mappings'][col] or confidence > self.rules['word_mappings'][col][word]['confidence']:
                            self.rules['word_mappings'][col][word] = {
                                'value': value,
                                'confidence': round(confidence, 3),
                                'occurrences': count
                            }
    
    def _discover_conditional_rules(self, data):
        """Discover if-then rules like 'if TEMP then measureType=Temperature'"""
        print("   Discovering conditional rules...")
        
        # Clear previous rules
        self.rules['conditional_rules'] = []
        
        # Temperature rules
        temp_keywords = ['TEMP', 'TMP', 'TEMPERATURE']
        temp_rows = [r for r in data if any(kw in self._safe_str(r.get('description', '')).upper() for kw in temp_keywords)]
        if temp_rows:
            temp_type_matches = sum(1 for r in temp_rows if self._safe_compare(r.get('measureType'), 'Temperature'))
            if len(temp_rows) > 5 and temp_type_matches / len(temp_rows) > 0.9:
                self.rules['conditional_rules'].append({
                    'condition': {'description_contains': temp_keywords},
                    'then': {'measureType': 'Temperature'},
                    'confidence': temp_type_matches / len(temp_rows),
                    'priority': 1
                })
        
        # Pressure rules
        press_keywords = ['PRESS', 'PRESSURE', 'PR']
        press_rows = [r for r in data if any(kw in self._safe_str(r.get('description', '')).upper() for kw in press_keywords)]
        if press_rows:
            press_type_matches = sum(1 for r in press_rows if self._safe_compare(r.get('measureType'), 'Pressure'))
            if len(press_rows) > 5 and press_type_matches / len(press_rows) > 0.9:
                self.rules['conditional_rules'].append({
                    'condition': {'description_contains': press_keywords},
                    'then': {'measureType': 'Pressure'},
                    'confidence': press_type_matches / len(press_rows),
                    'priority': 1
                })
        
        # NEW: Location rules based on common errors
        self._discover_location_rules(data)
    
    def _discover_location_rules(self, data):
        """Discover rules for measureLocation based on common patterns"""
        print("   Discovering location rules...")
        
        # I/L -> Inlet
        il_rows = [r for r in data if 'I/L' in self._safe_str(r.get('description', '')).upper()]
        if il_rows:
            il_matches = sum(1 for r in il_rows if self._safe_compare(r.get('measureLocation'), 'Inlet'))
            if len(il_rows) > 3 and il_matches / len(il_rows) > 0.8:
                self.rules['conditional_rules'].append({
                    'condition': {'description_contains': ['I/L']},
                    'then': {'measureLocation': 'Inlet', 'measureLocationName': 'Inlet'},
                    'confidence': il_matches / len(il_rows),
                    'priority': 2
                })
        
        # O/L -> Outlet
        ol_rows = [r for r in data if 'O/L' in self._safe_str(r.get('description', '')).upper()]
        if ol_rows:
            ol_matches = sum(1 for r in ol_rows if self._safe_compare(r.get('measureLocation'), 'Outlet'))
            if len(ol_rows) > 3 and ol_matches / len(ol_rows) > 0.8:
                self.rules['conditional_rules'].append({
                    'condition': {'description_contains': ['O/L']},
                    'then': {'measureLocation': 'Outlet', 'measureLocationName': 'Outlet'},
                    'confidence': ol_matches / len(ol_rows),
                    'priority': 2
                })
    
    def _discover_priority_rules(self, data):
        """Discover priority rules like 'temp %' -> '% has priority'"""
        print("   Discovering priority rules...")
        
        # Clear previous rules
        self.rules['priority_rules'] = []
        
        # Percentage over temperature
        self.rules['priority_rules'].append({
            'name': 'percentage_over_temperature',
            'conditions': [
                {'description_contains': ['TEMP', '%']},
                {'description_contains': ['PERCENT']}
            ],
            'action': {'measureUnit': '%', 'measureType': 'Percentage'},
            'priority': 10
        })
    
    def _discover_hierarchical_patterns(self, data):
        """Discover equipment -> system -> component hierarchies"""
        print("   Discovering hierarchical patterns...")
        
        # Clear previous patterns
        self.rules['hierarchical_patterns'] = {}
        
        # Group by equipment and analyze system patterns
        equipment_systems = defaultdict(Counter)
        for row in data:
            equipment = self._safe_str(row.get('equipment'))
            system = self._safe_str(row.get('system'))
            if equipment and system and equipment != 'Unassigned' and system != 'Unassigned':
                equipment_systems[equipment][system] += 1
        
        for equipment, systems in equipment_systems.items():
            if len(systems) == 1:  # Consistent mapping
                system = systems.most_common(1)[0][0]
                self.rules['hierarchical_patterns'][equipment] = {'system': system}
    
    def _discover_negative_patterns(self, data):
        """Discover patterns that should NOT be applied (based on common errors)"""
        print("   Discovering negative patterns...")
        
        # Clear previous negative patterns
        self.rules['negative_patterns'] = {}
        
        # Analyze cases where certain patterns lead to wrong predictions
        # For now, we'll add some common error patterns based on your verification results
        
        # Don't predict 'Bearing' for subcomponent if description doesn't have clear bearing context
        self.rules['negative_patterns']['subcomponent'] = {
            'conditions': [
                {'description_contains': ['TEMP', 'PRES']},  # Temperature/pressure readings
                {'description_not_contains': ['BRG', 'BEARING']}  # But no bearing context
            ],
            'action': 'skip_prediction'
        }
    
    def _discover_measurement_relationships(self, data):
        """Discover relationships between measurement types and units"""
        print("   Discovering measurement relationships...")
        
        # Temperature -> Degc
        temp_rows = [r for r in data if self._safe_compare(r.get('measureType'), 'Temperature')]
        if temp_rows:
            degc_matches = sum(1 for r in temp_rows if self._safe_compare(r.get('measureUnit'), 'Degc'))
            if len(temp_rows) > 10 and degc_matches / len(temp_rows) > 0.9:
                self.rules['column_relationships']['measureUnit_temp'] = {
                    'type': 'conditional_default',
                    'condition': {'measureType': 'Temperature'},
                    'value': 'Degc',
                    'confidence': degc_matches / len(temp_rows)
                }
        
        # Pressure -> Kg/Cm2
        pressure_rows = [r for r in data if self._safe_compare(r.get('measureType'), 'Pressure')]
        if pressure_rows:
            pressure_unit_matches = sum(1 for r in pressure_rows if self._safe_compare(r.get('measureUnit'), 'Kg/Cm2'))
            if len(pressure_rows) > 10 and pressure_unit_matches / len(pressure_rows) > 0.8:
                self.rules['column_relationships']['measureUnit_pressure'] = {
                    'type': 'conditional_default',
                    'condition': {'measureType': 'Pressure'},
                    'value': 'Kg/Cm2',
                    'confidence': pressure_unit_matches / len(pressure_rows)
                }
    
    def _analyze_plus_relationship(self, data, source_col, target_col):
        """Analyze + relationships (minor modifications)"""
        matches = []
        for row in data:
            source = row.get(source_col)
            target = row.get(target_col)
            
            # Convert to strings safely
            source_str = self._safe_str(source)
            target_str = self._safe_str(target)
            
            if source_str and target_str and source_str != target_str:
                # Check if target is source with minor modifications
                try:
                    if target_str.startswith(source_str) or source_str in target_str:
                        matches.append((source_str, target_str))
                except (TypeError, AttributeError) as e:
                    # Skip this row if there's a type error
                    continue
        
        if len(matches) > 10:  # Significant pattern
            return {
                'type': 'plus', 
                'source': source_col, 
                'confidence': len(matches)/len(data),
                'notation': '+'
            }
        return None
    
    def _analyze_instance_relationships(self, data):
        """Analyze instance columns (usually "1" or "1.0")"""
        instance_columns = ['systemInstance', 'equipmentInstance', 'componentInstance', 
                           'subcomponentInstance', 'measureLocationInstance']
        
        for col in instance_columns:
            values = [self._safe_str(row.get(col)) for row in data if row.get(col) and pd.notna(row.get(col))]
            if values:
                most_common = Counter(values).most_common(1)[0]
                if most_common[1] / len(values) > 0.8:  # 80% are the same value
                    self.rules['column_relationships'][col] = {
                        'type': 'default_value', 
                        'value': most_common[0],
                        'confidence': most_common[1] / len(values)
                    }
    
    def _safe_str(self, value):
        """Safely convert any value to string"""
        if pd.isna(value) or value is None:
            return ""
        if isinstance(value, (int, float)):
            # Convert 1.0 to "1" but keep other numbers as string
            if value == 1.0:
                return "1"
            elif value == int(value):
                return str(int(value))
            else:
                return str(value)
        return str(value)
    
    def _safe_compare(self, val1, val2):
        """Safely compare two values that might be different types"""
        return self._safe_str(val1) == self._safe_str(val2)
    
    def _load_all_training_data(self, training_dir):
        """Load all training data into memory"""
        all_data = []
        excel_files = [f for f in os.listdir(training_dir) if f.endswith(('.xlsx', '.xls'))]
        
        for file in excel_files:
            file_path = os.path.join(training_dir, file)
            try:
                df = pd.read_excel(file_path)
                print(f"      Loaded {file}: {len(df)} rows, {len(df.columns)} columns")
                all_data.extend(df.to_dict('records'))
            except Exception as e:
                print(f"      Error loading {file}: {e}")
            
        return all_data

    def save_rules(self, output_path):
        """Save discovered rules to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.rules, f, indent=2)
        print(f"Rules saved to: {output_path}")

def main():
    thinker = PatternThinker()
    training_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'training')
    
    print(f"Loading training data from: {training_dir}")
    rules = thinker.analyze_training_data(training_dir)
    
    # Save rules
    rules_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'pattern_rules.json')
    thinker.save_rules(rules_file)
    
    # Print summary
    print(f"DISCOVERED RULES SUMMARY:")
    print(f"   Column Relationships: {len(rules['column_relationships'])}")
    print(f"   Word Mappings: {sum(len(v) for v in rules['word_mappings'].values())}")
    print(f"   Conditional Rules: {len(rules['conditional_rules'])}")
    print(f"   Priority Rules: {len(rules['priority_rules'])}")
    print(f"   Hierarchical Patterns: {len(rules['hierarchical_patterns'])}")
    print(f"   Negative Patterns: {len(rules['negative_patterns'])}")

if __name__ == "__main__":
    main()