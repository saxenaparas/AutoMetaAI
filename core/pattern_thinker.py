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
            'hierarchical_patterns': {}
        }
    
    def analyze_training_data(self, training_dir):
        """Deep analysis to discover complex patterns"""
        print("THINKING: Analyzing training data for patterns...")
        
        # Load all training files
        all_data = self._load_all_training_data(training_dir)
        
        # Discover different types of patterns
        self._discover_column_relationships(all_data)
        self._discover_word_mappings(all_data) 
        self._discover_conditional_rules(all_data)
        self._discover_priority_rules(all_data)
        self._discover_hierarchical_patterns(all_data)
        
        return self.rules
    
    def _discover_column_relationships(self, data):
        """Discover * and + relationships between columns"""
        print("   Analyzing column relationships...")
        
        # systemName = system *
        system_matches = sum(1 for row in data if self._safe_compare(row.get('system'), row.get('systemName')))
        system_total = sum(1 for row in data if row.get('system') and row.get('systemName'))
        if system_total > 0 and system_matches / system_total > 0.8:
            self.rules['column_relationships']['systemName'] = {'type': 'equals', 'source': 'system', 'confidence': system_matches/system_total}
        
        # equipmentName = equipment +
        equipment_plus = self._analyze_plus_relationship(data, 'equipment', 'equipmentName')
        if equipment_plus:
            self.rules['column_relationships']['equipmentName'] = equipment_plus
            
        # Add instance relationships (usually "1" or "1.0")
        self._analyze_instance_relationships(data)
    
    def _discover_word_mappings(self, data):
        """Discover word -> value mappings with confidence"""
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
        
        # Convert to confidence scores
        for (col, word), counts in word_mappings.items():
            total = sum(counts.values())
            if total >= 2:  # Minimum occurrences
                best_value, best_count = counts.most_common(1)[0]
                confidence = best_count / total
                if confidence >= 0.7:
                    if col not in self.rules['word_mappings']:
                        self.rules['word_mappings'][col] = {}
                    self.rules['word_mappings'][col][word] = {
                        'value': best_value,
                        'confidence': confidence,
                        'occurrences': best_count
                    }
    
    def _discover_conditional_rules(self, data):
        """Discover if-then rules like 'if TEMP then measureType=Temperature'"""
        print("   Discovering conditional rules...")
        
        # Temperature rules
        temp_rows = [r for r in data if 'TEMP' in self._safe_str(r.get('description', '')).upper()]
        if temp_rows:
            temp_type_matches = sum(1 for r in temp_rows if self._safe_compare(r.get('measureType'), 'Temperature'))
            if len(temp_rows) > 5 and temp_type_matches / len(temp_rows) > 0.9:
                self.rules['conditional_rules'].append({
                    'condition': {'description_contains': ['TEMP', 'TMP', 'TEMPERATURE']},
                    'then': {'measureType': 'Temperature'},
                    'confidence': temp_type_matches / len(temp_rows),
                    'priority': 1
                })
        
        # Pressure rules
        press_rows = [r for r in data if any(word in self._safe_str(r.get('description', '')).upper() 
                                           for word in ['PRESS', 'PRESSURE', 'PR'])]
        if press_rows:
            press_type_matches = sum(1 for r in press_rows if self._safe_compare(r.get('measureType'), 'Pressure'))
            if len(press_rows) > 5 and press_type_matches / len(press_rows) > 0.9:
                self.rules['conditional_rules'].append({
                    'condition': {'description_contains': ['PRESS', 'PRESSURE', 'PR']},
                    'then': {'measureType': 'Pressure'},
                    'confidence': press_type_matches / len(press_rows),
                    'priority': 1
                })
    
    def _discover_priority_rules(self, data):
        """Discover priority rules like 'temp %' -> '% has priority'"""
        print("   Discovering priority rules...")
        
        # This is a simplified version - would need more complex analysis
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
            return {'type': 'plus', 'source': source_col, 'confidence': len(matches)/len(data)}
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

if __name__ == "__main__":
    main()