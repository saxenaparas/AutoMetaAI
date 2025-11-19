import pandas as pd
import json
import os
import sys
from datetime import datetime

# Add the core directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

from pattern_thinker import PatternThinker
from rule_based_predictor import RuleBasedPredictor
from verifier import PredictionVerifier

class ImprovementEngine:
    def __init__(self):
        self.iteration = 0
        self.history = []
        self.best_accuracy = 0
        self.best_rules = None
        
    def run_thinking_mode(self, max_iterations=5, target_accuracy=90):
        """Run iterative thinking and improvement loop"""
        print("STARTING THINKING & IMPROVEMENT MODE")
        print("=" * 60)
        
        while self.iteration < max_iterations:
            self.iteration += 1
            print(f"ITERATION {self.iteration}")
            print("-" * 40)
            
            # Step 1: Think - Discover new patterns
            thinking_result = self._run_thinking_step()
            if not thinking_result:
                print("Thinking step failed")
                break
            
            # Step 2: Predict - Apply new rules
            prediction_result = self._run_prediction_step()
            if not prediction_result:
                print("Prediction step failed")
                break
            
            # Step 3: Verify - Check against ground truth
            verification_result = self._run_verification_step()
            if not verification_result:
                print("Verification step failed")
                break
            
            # Step 4: Learn - Analyze results and plan improvements
            improvement_plan = self._analyze_improvements(verification_result)
            
            # Step 5: Record progress
            self._record_iteration(verification_result, improvement_plan)
            
            # Check if target achieved
            if verification_result['overall_accuracy'] >= target_accuracy:
                print(f"TARGET ACCURACY {target_accuracy}% ACHIEVED!")
                break
        
        self._print_final_report()
    
    def _run_thinking_step(self):
        """Run pattern thinker to discover new rules"""
        print("   STEP 1: Thinking...")
        try:
            thinker = PatternThinker()
            training_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'training')
            rules = thinker.analyze_training_data(training_dir)
            
            # Save rules
            rules_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'pattern_rules.json')
            thinker.save_rules(rules_file)
            
            print("   Thinking completed")
            return True
            
        except Exception as e:
            print(f"   Thinking error: {e}")
            return False
    
    def _run_prediction_step(self):
        """Run rule-based predictor with new rules"""
        print("   STEP 2: Predicting...")
        try:
            predictor = RuleBasedPredictor()
            
            # Process all files in input directory
            input_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'input')
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
            
            # Create directories if they don't exist
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Find all Excel files in input directory
            import glob
            excel_files = glob.glob(os.path.join(input_dir, "*.xlsx")) + glob.glob(os.path.join(input_dir, "*.xls"))
            
            if not excel_files:
                print("   No Excel files found in input directory!")
                return False
            
            for input_file in excel_files:
                filename = os.path.basename(input_file)
                output_file = os.path.join(output_dir, f"IMPROVED_PREDICTED_{filename}")
                print(f"   Processing: {filename}")
                predictor.predict_single_file(input_file, output_file)
            
            print("   Prediction completed")
            return True
            
        except Exception as e:
            print(f"   Prediction error: {e}")
            return False
    
    def _run_verification_step(self):
        """Run verifier to check accuracy"""
        print("   STEP 3: Verifying...")
        try:
            verifier = PredictionVerifier()
            
            # File paths - use the new improved predictions
            predicted_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'IMPROVED_PREDICTED_sample_data.xlsx')
            verified_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'verification', 'Filled_Sample_Data_For_Verification.xlsx')
            
            # Check if files exist
            if not os.path.exists(predicted_file):
                print(f"   Predicted file not found: {predicted_file}")
                return None
                
            if not os.path.exists(verified_file):
                print(f"   Verified file not found: {verified_file}")
                return None
            
            # Run verification
            accuracy = verifier.verify_predictions(predicted_file, verified_file)
            
            if accuracy is not None:
                print(f"   Verification completed - Accuracy: {accuracy}%")
                return {
                    'overall_accuracy': accuracy,
                    'timestamp': datetime.now()
                }
            else:
                print("   Could not determine accuracy from verification")
                return None
                
        except Exception as e:
            print(f"   Verification error: {e}")
            return None
    
    def _analyze_improvements(self, verification_result):
        """Analyze what needs improvement"""
        print("   STEP 4: Learning...")
        
        accuracy = verification_result['overall_accuracy']
        
        if accuracy < 50:
            improvement_plan = {
                'focus': 'Basic patterns',
                'actions': ['Add missing high-frequency word mappings', 'Fix column relationships', 'Add basic conditional rules']
            }
        elif accuracy < 70:
            improvement_plan = {
                'focus': 'Intermediate patterns', 
                'actions': ['Refine conditional rules', 'Add priority rules', 'Improve hierarchical patterns']
            }
        else:
            improvement_plan = {
                'focus': 'Advanced patterns',
                'actions': ['Optimize hierarchical patterns', 'Fine-tune confidence thresholds', 'Add complex conditional logic']
            }
        
        print(f"   Learning completed - Focus: {improvement_plan['focus']}")
        return improvement_plan
    
    def _record_iteration(self, verification_result, improvement_plan):
        """Record iteration results"""
        iteration_data = {
            'iteration': self.iteration,
            'accuracy': verification_result['overall_accuracy'],
            'timestamp': verification_result['timestamp'],
            'improvement_plan': improvement_plan,
            'improvement_from_start': verification_result['overall_accuracy'] - (self.history[0]['accuracy'] if self.history else 0)
        }
        
        self.history.append(iteration_data)
        
        # Update best accuracy
        if verification_result['overall_accuracy'] > self.best_accuracy:
            self.best_accuracy = verification_result['overall_accuracy']
        
        print(f"   Iteration {self.iteration} recorded: {verification_result['overall_accuracy']}% accuracy")
    
    def _print_final_report(self):
        """Print final improvement report"""
        print("\n" + "=" * 60)
        print("FINAL IMPROVEMENT REPORT")
        print("=" * 60)
        
        if not self.history:
            print("No iterations completed")
            return
        
        initial_acc = self.history[0]['accuracy']
        final_acc = self.history[-1]['accuracy']
        improvement = final_acc - initial_acc
        
        print(f"Initial Accuracy: {initial_acc:.1f}%")
        print(f"Final Accuracy: {final_acc:.1f}%")
        print(f"Total Improvement: {improvement:+.1f}%")
        print(f"Best Accuracy: {self.best_accuracy:.1f}%")
        print(f"Iterations Completed: {len(self.history)}")
        
        print(f"ITERATION HISTORY:")
        for iter_data in self.history:
            print(f"   Iteration {iter_data['iteration']}: {iter_data['accuracy']:.1f}% "
                  f"(Δ{iter_data['improvement_from_start']:+.1f}%)")

def main():
    engine = ImprovementEngine()
    engine.run_thinking_mode(max_iterations=3, target_accuracy=80)

if __name__ == "__main__":
    main()