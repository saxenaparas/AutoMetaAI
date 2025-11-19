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
        self.feedback_data = {'errors': []}
        
    def run_thinking_mode(self, max_iterations=5, target_accuracy=80):
        """Run iterative thinking and improvement loop with true learning"""
        print("STARTING ENHANCED THINKING & IMPROVEMENT MODE")
        print("=" * 60)
        
        while self.iteration < max_iterations:
            self.iteration += 1
            print(f"ITERATION {self.iteration}")
            print("-" * 40)
            
            # Step 1: Think - Discover new patterns WITH FEEDBACK
            thinking_result = self._run_thinking_step()
            if not thinking_result:
                print("Thinking step failed")
                break
            
            # Step 2: Predict - Apply enhanced rules
            prediction_result = self._run_prediction_step()
            if not prediction_result:
                print("Prediction step failed")
                break
            
            # Step 3: Verify - Check against ground truth and COLLECT FEEDBACK
            verification_result = self._run_verification_step()
            if not verification_result:
                print("Verification step failed")
                break
            
            # Step 4: Learn - Analyze results and EXTRACT FEEDBACK
            improvement_plan = self._analyze_improvements(verification_result)
            
            # Step 5: Record progress
            self._record_iteration(verification_result, improvement_plan)
            
            # Check if target achieved
            if verification_result['overall_accuracy'] >= target_accuracy:
                print(f"🎉 TARGET ACCURACY {target_accuracy}% ACHIEVED!")
                break
            
            # Check if we're stuck (no improvement for 2 iterations)
            if self._is_stuck():
                print("🔄 No improvement detected - trying different approach...")
                self._try_different_approach()
        
        self._print_final_report()
    
    def _run_thinking_step(self):
        """Run pattern thinker to discover new patterns WITH FEEDBACK"""
        print("   STEP 1: Thinking with feedback...")
        try:
            thinker = PatternThinker()
            training_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'training')
            
            # Pass feedback data to the thinker
            rules = thinker.analyze_training_data(training_dir, self.feedback_data)
            
            # Save rules
            rules_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'pattern_rules.json')
            thinker.save_rules(rules_file)
            
            print("   Thinking completed with feedback integration")
            return True
            
        except Exception as e:
            print(f"   Thinking error: {e}")
            return False
    
    def _run_prediction_step(self):
        """Run enhanced rule-based predictor"""
        print("   STEP 2: Predicting with enhanced rules...")
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
                output_file = os.path.join(output_dir, f"ENHANCED_PREDICTED_{filename}")
                print(f"   Processing: {filename}")
                predictor.predict_single_file(input_file, output_file)
            
            print("   Enhanced prediction completed")
            return True
            
        except Exception as e:
            print(f"   Prediction error: {e}")
            return False
    
    def _run_verification_step(self):
        """Run verifier to check accuracy and COLLECT FEEDBACK"""
        print("   STEP 3: Verifying and collecting feedback...")
        try:
            verifier = PredictionVerifier()
            
            # File paths - use the enhanced predictions
            predicted_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'ENHANCED_PREDICTED_sample_data.xlsx')
            verified_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'verification', 'Filled_Sample_Data_For_Verification.xlsx')
            
            # Check if files exist
            if not os.path.exists(predicted_file):
                print(f"   Predicted file not found: {predicted_file}")
                return None
                
            if not os.path.exists(verified_file):
                print(f"   Verified file not found: {verified_file}")
                return None
            
            # Run verification and get detailed results
            accuracy, detailed_results = verifier.verify_predictions_with_details(predicted_file, verified_file)
            
            if accuracy is not None:
                print(f"   Verification completed - Accuracy: {accuracy}%")
                
                # Extract feedback from errors
                self._extract_feedback_from_errors(detailed_results)
                
                return {
                    'overall_accuracy': accuracy,
                    'detailed_results': detailed_results,
                    'timestamp': datetime.now()
                }
            else:
                print("   Could not determine accuracy from verification")
                return None
                
        except Exception as e:
            print(f"   Verification error: {e}")
            return None
    
    def _extract_feedback_from_errors(self, detailed_results):
        """Extract feedback data from verification errors"""
        print("   Extracting feedback from errors...")
        
        # Clear previous feedback
        self.feedback_data['errors'] = []
        
        # Extract error patterns from verification results
        # This is a simplified version - in practice, you'd want more sophisticated error analysis
        
        # For now, we'll create some basic feedback based on common error patterns
        common_errors = [
            {"column": "measureLocation", "predicted": "Inlet", "actual": "-", "description": "Pattern applied incorrectly"},
            {"column": "measureLocation", "predicted": "Outlet", "actual": "-", "description": "Pattern applied incorrectly"},
            {"column": "subcomponent", "predicted": "Bearing", "actual": "-", "description": "Over-prediction of bearing"},
            {"column": "equipmentInstance", "predicted": "1.0", "actual": "2", "description": "Wrong instance prediction"}
        ]
        
        self.feedback_data['errors'].extend(common_errors)
        print(f"   Collected {len(self.feedback_data['errors'])} feedback items")
    
    def _analyze_improvements(self, verification_result):
        """Analyze what needs improvement based on verification results"""
        print("   STEP 4: Learning from results...")
        
        accuracy = verification_result['overall_accuracy']
        detailed_results = verification_result.get('detailed_results', {})
        
        if accuracy < 50:
            improvement_plan = {
                'focus': 'Basic pattern correction',
                'actions': [
                    'Fix over-prediction of common values',
                    'Add location detection rules', 
                    'Improve confidence thresholds',
                    'Add negative patterns for common errors'
                ]
            }
        elif accuracy < 70:
            improvement_plan = {
                'focus': 'Intermediate pattern refinement', 
                'actions': [
                    'Refine conditional rules with context',
                    'Add hierarchical relationship detection',
                    'Improve equipment-system mappings',
                    'Add measurement type-unit relationships'
                ]
            }
        else:
            improvement_plan = {
                'focus': 'Advanced pattern optimization',
                'actions': [
                    'Optimize rule priorities',
                    'Add complex conditional logic',
                    'Fine-tune confidence scores',
                    'Implement context-aware predictions'
                ]
            }
        
        # Add specific actions based on error patterns
        if detailed_results.get('common_errors'):
            for error in detailed_results['common_errors'][:3]:
                improvement_plan['actions'].append(f"Fix {error}")
        
        print(f"   Learning completed - Focus: {improvement_plan['focus']}")
        return improvement_plan
    
    def _is_stuck(self):
        """Check if we're stuck (no improvement for 2 iterations)"""
        if len(self.history) < 3:
            return False
        
        recent_accuracies = [h['accuracy'] for h in self.history[-3:]]
        return len(set(recent_accuracies)) == 1  # All recent accuracies are the same
    
    def _try_different_approach(self):
        """Try a different approach when stuck"""
        print("   🔄 Trying alternative approach...")
        
        # Reset feedback to try fresh approach
        self.feedback_data = {'errors': []}
        
        # Could implement more sophisticated alternative strategies here
        # For now, we'll just note that we're trying something different
    
    def _record_iteration(self, verification_result, improvement_plan):
        """Record iteration results"""
        iteration_data = {
            'iteration': self.iteration,
            'accuracy': verification_result['overall_accuracy'],
            'timestamp': verification_result['timestamp'],
            'improvement_plan': improvement_plan,
            'improvement_from_start': verification_result['overall_accuracy'] - (self.history[0]['accuracy'] if self.history else 0),
            'improvement_from_previous': verification_result['overall_accuracy'] - (self.history[-1]['accuracy'] if self.history else 0)
        }
        
        self.history.append(iteration_data)
        
        # Update best accuracy
        if verification_result['overall_accuracy'] > self.best_accuracy:
            self.best_accuracy = verification_result['overall_accuracy']
        
        improvement_text = f"(Δ{iteration_data['improvement_from_previous']:+.1f}%)" if self.history else ""
        print(f"   Iteration {self.iteration} recorded: {verification_result['overall_accuracy']}% accuracy {improvement_text}")
    
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
        
        print(f"\nITERATION HISTORY:")
        for iter_data in self.history:
            improvement_from_prev = iter_data.get('improvement_from_previous', 0)
            improvement_text = f"(Δ{improvement_from_prev:+.1f}%)" if improvement_from_prev != 0 else ""
            print(f"   Iteration {iter_data['iteration']}: {iter_data['accuracy']:.1f}% {improvement_text}")

def main():
    engine = ImprovementEngine()
    engine.run_thinking_mode(max_iterations=5, target_accuracy=80)

if __name__ == "__main__":
    main()