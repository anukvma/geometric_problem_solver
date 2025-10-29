"""
Comprehensive Evaluation Script for Geometric Problem Solver (Gemini Version)
Evaluates the solver on a dataset with multiple choice questions
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import csv
from collections import defaultdict
import time
import random

from geometric_problem_solver import ScalableGeometricProblemSolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeometricSolverEvaluator:
    """Evaluator for the geometric problem solver"""

    def __init__(self, config: Dict[str, str], dataset_path: str, include_image: bool = False):
        """
        Initialize evaluator

        Args:
            config: Configuration dict with API keys and Neo4j credentials
            dataset_path: Path to the dataset folder
            include_image: If True, includes image along with knowledge graph in prompts
        """
        self.config = config
        self.dataset_path = Path(dataset_path)
        self.include_image = include_image
        self.solver = ScalableGeometricProblemSolver(config)
        self.results = []
        self.evaluation_metrics = {
            'total_problems': 0,
            'successful_solves': 0,
            'correct_answers': 0,
            'failed_solves': 0,
            'accuracy': 0.0,
            'success_rate': 0.0,
            'image_included': include_image
        }

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load all problems from the dataset"""

        logger.info(f"Loading dataset from: {self.dataset_path}")
        problems = []

        # Iterate through all subdirectories
        for problem_dir in sorted(self.dataset_path.iterdir()):
            if not problem_dir.is_dir():
                continue

            # Check for required files
            image_path = problem_dir / "img_diagram.png"
            data_path = problem_dir / "data.json"

            if not image_path.exists() or not data_path.exists():
                logger.warning(f"Skipping {problem_dir.name}: Missing required files")
                continue

            try:
                # Load problem data
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract required fields
                problem = {
                    'problem_id': problem_dir.name,
                    'image_path': str(image_path),
                    'problem_text': data.get('compact_text', ''),
                    'choices': data.get('choices', []),
                    'ground_truth': data.get('answer', ''),
                    'full_data': data  # Keep full data for reference
                }

                problems.append(problem)
                logger.info(f"Loaded problem {problem_dir.name}")

            except Exception as e:
                logger.error(f"Error loading {problem_dir.name}: {e}")
                continue

        logger.info(f"Loaded {len(problems)} problems from dataset")
        return problems

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        if not answer:
            return ""

        # Remove whitespace and convert to lowercase
        normalized = str(answer).strip().lower()

        # Remove common units and symbols
        normalized = normalized.replace('°', '').replace('degrees', '')
        normalized = normalized.replace('units', '').replace('unit', '')

        return normalized.strip()

    def compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """Compare predicted answer with ground truth"""

        pred_norm = self.normalize_answer(predicted)
        gt_norm = self.normalize_answer(ground_truth)

        # Direct match
        if pred_norm == gt_norm:
            return True

        # Try to extract numeric values and compare
        try:
            pred_num = float(pred_norm)
            gt_num = float(gt_norm)
            # Allow small floating point differences
            return abs(pred_num - gt_num) < 0.01
        except (ValueError, TypeError):
            pass

        # Check if predicted is a substring of ground truth or vice versa
        if pred_norm in gt_norm or gt_norm in pred_norm:
            return True

        return False

    def evaluate_single_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate solver on a single problem"""

        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating Problem: {problem['problem_id']}")
        logger.info(f"{'='*80}")

        start_time = time.time()

        try:
            # Solve the problem
            result = self.solver.solve_geometric_problem(
                image_path=problem['image_path'],
                problem_text=problem['problem_text'],
                problem_id=problem['problem_id'],
                choices=problem['choices'] if problem['choices'] else None,
                include_image=self.include_image
            )

            solve_time = time.time() - start_time

            # Extract predicted answer
            predicted_choice = result.get('selected_choice', '')
            predicted_value = result.get('selected_choice_value', '')

            # If no choice selected, try to extract from found_variables
            if not predicted_choice and result.get('found_variables'):
                # Get the first found variable value
                first_var = list(result.get('found_variables', {}).values())
                if first_var:
                    predicted_value = str(first_var[0].get('value', ''))

            # Determine correctness
            is_correct = False
            if predicted_choice:
                # For multiple choice, compare the choice letter
                is_correct = self.compare_answers(predicted_choice, problem['ground_truth'])
            elif predicted_value:
                # Compare the value directly
                is_correct = self.compare_answers(predicted_value, problem['ground_truth'])

            # Compile evaluation result
            eval_result = {
                'problem_id': problem['problem_id'],
                'problem_text': problem['problem_text'],
                'choices': problem['choices'],
                'ground_truth': problem['ground_truth'],
                'predicted_choice': predicted_choice,
                'predicted_value': predicted_value,
                'is_correct': is_correct,
                'success': result.get('success', False),
                'confidence': result.get('confidence', 0.0),
                'total_steps': result.get('total_steps', 0),
                'solve_time': solve_time,
                'explanation': result.get('explanation', ''),
                'error': None,
                'full_result': result
            }

            logger.info(f"Ground Truth: {problem['ground_truth']}")
            logger.info(f"Predicted Choice: {predicted_choice}")
            logger.info(f"Predicted Value: {predicted_value}")
            logger.info(f"Correct: {is_correct}")
            logger.info(f"Solve Time: {solve_time:.2f}s")

            return eval_result

        except Exception as e:
            logger.error(f"Error evaluating problem {problem['problem_id']}: {e}")

            return {
                'problem_id': problem['problem_id'],
                'problem_text': problem['problem_text'],
                'choices': problem['choices'],
                'ground_truth': problem['ground_truth'],
                'predicted_choice': '',
                'predicted_value': '',
                'is_correct': False,
                'success': False,
                'confidence': 0.0,
                'total_steps': 0,
                'solve_time': time.time() - start_time,
                'explanation': '',
                'error': str(e),
                'full_result': None
            }

    def evaluate_all(self, max_problems: Optional[int] = None, random_seed: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate solver on all problems in the dataset

        Args:
            max_problems: Maximum number of problems to evaluate
            random_seed: Random seed for reproducible problem selection
        """

        logger.info("Starting comprehensive evaluation...")

        # Load dataset
        problems = self.load_dataset()

        if max_problems:
            # Randomly select problems instead of taking first N
            if random_seed is not None:
                random.seed(random_seed)
            problems = random.sample(problems, min(max_problems, len(problems)))
            logger.info(f"Randomly selected {len(problems)} problems from dataset (seed: {random_seed})")

        total_problems = len(problems)
        self.evaluation_metrics['total_problems'] = total_problems
        self.evaluation_metrics['random_seed'] = random_seed

        # Evaluate each problem
        for i, problem in enumerate(problems, 1):
            logger.info(f"\nProgress: {i}/{total_problems}")

            result = self.evaluate_single_problem(problem)
            self.results.append(result)

            # Update metrics
            if result['success']:
                self.evaluation_metrics['successful_solves'] += 1
            else:
                self.evaluation_metrics['failed_solves'] += 1

            if result['is_correct']:
                self.evaluation_metrics['correct_answers'] += 1

        # Calculate final metrics
        if total_problems > 0:
            self.evaluation_metrics['success_rate'] = (
                self.evaluation_metrics['successful_solves'] / total_problems * 100
            )
            self.evaluation_metrics['accuracy'] = (
                self.evaluation_metrics['correct_answers'] / total_problems * 100
            )

        # Add additional statistics
        self.evaluation_metrics['avg_confidence'] = sum(
            r['confidence'] for r in self.results
        ) / len(self.results) if self.results else 0.0

        self.evaluation_metrics['avg_solve_time'] = sum(
            r['solve_time'] for r in self.results
        ) / len(self.results) if self.results else 0.0

        self.evaluation_metrics['avg_steps'] = sum(
            r['total_steps'] for r in self.results if r['total_steps'] > 0
        ) / len([r for r in self.results if r['total_steps'] > 0]) if any(r['total_steps'] > 0 for r in self.results) else 0.0

        return self.evaluation_metrics

    def save_results(self, output_dir: str = "evaluation_results"):
        """Save evaluation results to files"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results as JSON
        json_path = output_path / f"evaluation_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'evaluation_metrics': self.evaluation_metrics,
                'detailed_results': [
                    {k: v for k, v in r.items() if k != 'full_result'}  # Exclude full result for readability
                    for r in self.results
                ]
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved detailed results to: {json_path}")

        # Save summary CSV
        csv_path = output_path / f"evaluation_summary_{timestamp}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'problem_id', 'ground_truth', 'predicted_choice', 'predicted_value',
                'is_correct', 'success', 'confidence', 'total_steps', 'solve_time', 'error'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                writer.writerow({
                    'problem_id': result['problem_id'],
                    'ground_truth': result['ground_truth'],
                    'predicted_choice': result['predicted_choice'],
                    'predicted_value': result['predicted_value'],
                    'is_correct': result['is_correct'],
                    'success': result['success'],
                    'confidence': f"{result['confidence']:.2%}",
                    'total_steps': result['total_steps'],
                    'solve_time': f"{result['solve_time']:.2f}s",
                    'error': result['error'] or ''
                })

        logger.info(f"Saved summary CSV to: {csv_path}")

        # Save metrics report
        report_path = output_path / f"evaluation_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("GEOMETRIC PROBLEM SOLVER - EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Path: {self.dataset_path}\n")
            f.write(f"Image Included in Prompt: {self.evaluation_metrics.get('image_included', False)}\n")
            if self.evaluation_metrics.get('random_seed') is not None:
                f.write(f"Random Seed: {self.evaluation_metrics.get('random_seed')} (for reproducible problem selection)\n")
            f.write("\n")

            f.write("OVERALL METRICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Problems:        {self.evaluation_metrics['total_problems']}\n")
            f.write(f"Successful Solves:     {self.evaluation_metrics['successful_solves']}\n")
            f.write(f"Failed Solves:         {self.evaluation_metrics['failed_solves']}\n")
            f.write(f"Correct Answers:       {self.evaluation_metrics['correct_answers']}\n")
            f.write(f"\n")
            f.write(f"Success Rate:          {self.evaluation_metrics['success_rate']:.2f}%\n")
            f.write(f"Accuracy:              {self.evaluation_metrics['accuracy']:.2f}%\n")
            f.write(f"Average Confidence:    {self.evaluation_metrics['avg_confidence']:.2%}\n")
            f.write(f"Average Solve Time:    {self.evaluation_metrics['avg_solve_time']:.2f}s\n")
            f.write(f"Average Steps:         {self.evaluation_metrics['avg_steps']:.1f}\n")
            f.write("\n")

            # Problem-by-problem breakdown
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 80 + "\n")

            for result in self.results:
                f.write(f"\nProblem ID: {result['problem_id']}\n")
                f.write(f"  Question: {result['problem_text'][:100]}...\n")
                if result['choices']:
                    f.write(f"  Choices: {result['choices']}\n")
                f.write(f"  Ground Truth: {result['ground_truth']}\n")
                f.write(f"  Predicted: {result['predicted_choice'] or result['predicted_value']}\n")
                f.write(f"  Correct: {'✓' if result['is_correct'] else '✗'}\n")
                f.write(f"  Success: {'✓' if result['success'] else '✗'}\n")
                f.write(f"  Confidence: {result['confidence']:.2%}\n")
                f.write(f"  Steps: {result['total_steps']}\n")
                f.write(f"  Time: {result['solve_time']:.2f}s\n")
                if result['error']:
                    f.write(f"  Error: {result['error']}\n")

        logger.info(f"Saved evaluation report to: {report_path}")

        return {
            'json_path': str(json_path),
            'csv_path': str(csv_path),
            'report_path': str(report_path)
        }

    def print_summary(self):
        """Print evaluation summary to console"""

        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"\nTotal Problems:        {self.evaluation_metrics['total_problems']}")
        if self.evaluation_metrics.get('random_seed') is not None:
            print(f"Random Seed:           {self.evaluation_metrics.get('random_seed')}")
        print(f"Successful Solves:     {self.evaluation_metrics['successful_solves']}")
        print(f"Failed Solves:         {self.evaluation_metrics['failed_solves']}")
        print(f"Correct Answers:       {self.evaluation_metrics['correct_answers']}")
        print(f"\nSuccess Rate:          {self.evaluation_metrics['success_rate']:.2f}%")
        print(f"Accuracy:              {self.evaluation_metrics['accuracy']:.2f}%")
        print(f"Average Confidence:    {self.evaluation_metrics['avg_confidence']:.2%}")
        print(f"Average Solve Time:    {self.evaluation_metrics['avg_solve_time']:.2f}s")
        print(f"Average Steps:         {self.evaluation_metrics['avg_steps']:.1f}")
        print("\n" + "="*80)

        # Show some example results
        print("\nSAMPLE RESULTS (First 5 problems):")
        print("-" * 80)

        for result in self.results[:5]:
            status = "✓ CORRECT" if result['is_correct'] else "✗ INCORRECT"
            print(f"\n{result['problem_id']}: {status}")
            print(f"  Ground Truth: {result['ground_truth']}")
            print(f"  Predicted:    {result['predicted_choice'] or result['predicted_value']}")
            print(f"  Time:         {result['solve_time']:.2f}s")

    def close(self):
        """Clean up resources"""
        self.solver.close()


class ComparisonEvaluator:
    """Evaluator that compares performance with and without image in prompt"""

    def __init__(self, config: Dict[str, str], dataset_path: str):
        """Initialize comparison evaluator"""
        self.config = config
        self.dataset_path = dataset_path

    def run_comparison(self, max_problems: Optional[int] = None, random_seed: Optional[int] = None) -> Dict[str, Any]:
        """Run evaluation both with and without image and compare results

        Args:
            max_problems: Maximum number of problems to evaluate
            random_seed: Random seed for reproducible problem selection (same problems used for both modes)
        """

        logger.info("="*80)
        logger.info("STARTING COMPARISON EVALUATION")
        logger.info("Mode 1: Knowledge Graph Only")
        logger.info("Mode 2: Knowledge Graph + Image")
        logger.info("="*80)

        # Use same random seed for both evaluations to ensure same problems are selected
        if random_seed is None:
            random_seed = random.randint(0, 999999)
        logger.info(f"Using random seed: {random_seed} (ensures same problems selected for both modes)")

        # Run evaluation WITHOUT image (knowledge graph only)
        logger.info("\n" + "="*80)
        logger.info("EVALUATING WITH KNOWLEDGE GRAPH ONLY")
        logger.info("="*80 + "\n")
        evaluator_kg_only = GeometricSolverEvaluator(self.config, self.dataset_path, include_image=False)
        metrics_kg_only = evaluator_kg_only.evaluate_all(max_problems=max_problems, random_seed=random_seed)
        results_kg_only = evaluator_kg_only.results.copy()
        evaluator_kg_only.close()

        # Run evaluation WITH image (knowledge graph + image) - using SAME random seed
        logger.info("\n" + "="*80)
        logger.info("EVALUATING WITH KNOWLEDGE GRAPH + IMAGE")
        logger.info("="*80 + "\n")
        evaluator_with_image = GeometricSolverEvaluator(self.config, self.dataset_path, include_image=True)
        metrics_with_image = evaluator_with_image.evaluate_all(max_problems=max_problems, random_seed=random_seed)
        results_with_image = evaluator_with_image.results.copy()
        evaluator_with_image.close()

        # Compare results
        comparison = self._compare_results(
            metrics_kg_only, results_kg_only,
            metrics_with_image, results_with_image
        )

        # Add random seed to comparison results for documentation
        comparison['random_seed'] = random_seed

        return comparison

    def _compare_results(self, metrics_kg: Dict, results_kg: List,
                        metrics_img: Dict, results_img: List) -> Dict[str, Any]:
        """Compare results between two evaluation modes"""

        comparison = {
            'kg_only_metrics': metrics_kg,
            'with_image_metrics': metrics_img,
            'comparison_summary': {},
            'per_problem_comparison': []
        }

        # Overall comparison
        comparison['comparison_summary'] = {
            'accuracy_difference': metrics_img['accuracy'] - metrics_kg['accuracy'],
            'success_rate_difference': metrics_img['success_rate'] - metrics_kg['success_rate'],
            'kg_only_accuracy': metrics_kg['accuracy'],
            'with_image_accuracy': metrics_img['accuracy'],
            'kg_only_success_rate': metrics_kg['success_rate'],
            'with_image_success_rate': metrics_img['success_rate'],
            'total_problems': metrics_kg['total_problems']
        }

        # Per-problem comparison
        for r_kg, r_img in zip(results_kg, results_img):
            problem_comp = {
                'problem_id': r_kg['problem_id'],
                'ground_truth': r_kg['ground_truth'],
                'kg_only_prediction': r_kg['predicted_choice'] or r_kg['predicted_value'],
                'with_image_prediction': r_img['predicted_choice'] or r_img['predicted_value'],
                'kg_only_correct': r_kg['is_correct'],
                'with_image_correct': r_img['is_correct'],
                'kg_only_confidence': r_kg['confidence'],
                'with_image_confidence': r_img['confidence'],
                'kg_only_solve_time': r_kg['solve_time'],
                'with_image_solve_time': r_img['solve_time'],
                'agreement': (r_kg['predicted_choice'] or r_kg['predicted_value']) ==
                            (r_img['predicted_choice'] or r_img['predicted_value']),
                'both_correct': r_kg['is_correct'] and r_img['is_correct'],
                'only_kg_correct': r_kg['is_correct'] and not r_img['is_correct'],
                'only_image_correct': not r_kg['is_correct'] and r_img['is_correct'],
                'both_incorrect': not r_kg['is_correct'] and not r_img['is_correct']
            }
            comparison['per_problem_comparison'].append(problem_comp)

        # Calculate agreement statistics
        agreement_count = sum(1 for p in comparison['per_problem_comparison'] if p['agreement'])
        both_correct_count = sum(1 for p in comparison['per_problem_comparison'] if p['both_correct'])
        only_kg_correct_count = sum(1 for p in comparison['per_problem_comparison'] if p['only_kg_correct'])
        only_image_correct_count = sum(1 for p in comparison['per_problem_comparison'] if p['only_image_correct'])

        comparison['comparison_summary']['agreement_rate'] = (
            agreement_count / len(comparison['per_problem_comparison']) * 100
            if comparison['per_problem_comparison'] else 0
        )
        comparison['comparison_summary']['both_correct_count'] = both_correct_count
        comparison['comparison_summary']['only_kg_correct_count'] = only_kg_correct_count
        comparison['comparison_summary']['only_image_correct_count'] = only_image_correct_count

        return comparison

    def save_comparison_results(self, comparison: Dict[str, Any], output_dir: str = "evaluation_results", random_seed: Optional[int] = None):
        """Save comparison results to files

        Args:
            comparison: Comparison results dictionary
            output_dir: Output directory for results
            random_seed: Random seed used for problem selection (for documentation)
        """

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed comparison as JSON
        json_path = output_path / f"comparison_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved comparison results to: {json_path}")

        # Save comparison report
        report_path = output_path / f"comparison_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("KNOWLEDGE GRAPH vs KNOWLEDGE GRAPH + IMAGE COMPARISON\n")
            f.write("="*80 + "\n\n")

            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if random_seed is not None:
                f.write(f"Random Seed: {random_seed} (for reproducible problem selection)\n")
            f.write("\n")

            summary = comparison['comparison_summary']

            f.write("OVERALL COMPARISON:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Problems:                {summary['total_problems']}\n\n")

            f.write("KNOWLEDGE GRAPH ONLY:\n")
            f.write(f"  Accuracy:                    {summary['kg_only_accuracy']:.2f}%\n")
            f.write(f"  Success Rate:                {summary['kg_only_success_rate']:.2f}%\n\n")

            f.write("KNOWLEDGE GRAPH + IMAGE:\n")
            f.write(f"  Accuracy:                    {summary['with_image_accuracy']:.2f}%\n")
            f.write(f"  Success Rate:                {summary['with_image_success_rate']:.2f}%\n\n")

            f.write("DIFFERENCES:\n")
            f.write(f"  Accuracy Improvement:        {summary['accuracy_difference']:+.2f}%\n")
            f.write(f"  Success Rate Improvement:    {summary['success_rate_difference']:+.2f}%\n\n")

            f.write("AGREEMENT ANALYSIS:\n")
            f.write(f"  Agreement Rate:              {summary['agreement_rate']:.2f}%\n")
            f.write(f"  Both Correct:                {summary['both_correct_count']}\n")
            f.write(f"  Only KG Correct:             {summary['only_kg_correct_count']}\n")
            f.write(f"  Only Image Correct:          {summary['only_image_correct_count']}\n\n")

            # Problem-by-problem comparison
            f.write("PROBLEM-BY-PROBLEM COMPARISON:\n")
            f.write("-" * 80 + "\n\n")

            for prob in comparison['per_problem_comparison']:
                f.write(f"Problem ID: {prob['problem_id']}\n")
                f.write(f"  Ground Truth:          {prob['ground_truth']}\n")
                f.write(f"  KG Only Prediction:    {prob['kg_only_prediction']} {'✓' if prob['kg_only_correct'] else '✗'}\n")
                f.write(f"  With Image Prediction: {prob['with_image_prediction']} {'✓' if prob['with_image_correct'] else '✗'}\n")
                f.write(f"  Predictions Agree:     {'Yes' if prob['agreement'] else 'No'}\n")

                if prob['only_kg_correct']:
                    f.write(f"  Note: Only KG correct (Image approach failed)\n")
                elif prob['only_image_correct']:
                    f.write(f"  Note: Only Image+KG correct (KG-only approach failed)\n")
                elif prob['both_incorrect']:
                    f.write(f"  Note: Both approaches incorrect\n")

                f.write("\n")

        logger.info(f"Saved comparison report to: {report_path}")

        return {
            'json_path': str(json_path),
            'report_path': str(report_path)
        }

    def print_comparison_summary(self, comparison: Dict[str, Any]):
        """Print comparison summary to console"""

        summary = comparison['comparison_summary']

        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"\nTotal Problems: {summary['total_problems']}")
        if comparison.get('random_seed') is not None:
            print(f"Random Seed: {comparison.get('random_seed')} (same problems used for both modes)")
        print()

        print("KNOWLEDGE GRAPH ONLY:")
        print(f"  Accuracy:      {summary['kg_only_accuracy']:.2f}%")
        print(f"  Success Rate:  {summary['kg_only_success_rate']:.2f}%\n")

        print("KNOWLEDGE GRAPH + IMAGE:")
        print(f"  Accuracy:      {summary['with_image_accuracy']:.2f}%")
        print(f"  Success Rate:  {summary['with_image_success_rate']:.2f}%\n")

        print("IMPROVEMENT WITH IMAGE:")
        print(f"  Accuracy:      {summary['accuracy_difference']:+.2f}%")
        print(f"  Success Rate:  {summary['success_rate_difference']:+.2f}%\n")

        print("AGREEMENT ANALYSIS:")
        print(f"  Agreement Rate:        {summary['agreement_rate']:.2f}%")
        print(f"  Both Correct:          {summary['both_correct_count']}")
        print(f"  Only KG Correct:       {summary['only_kg_correct_count']}")
        print(f"  Only Image Correct:    {summary['only_image_correct_count']}")
        print("\n" + "="*80)


def main_comparison():
    """Main function for comparison evaluation"""

    # Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = ""
    GEMINI_API_KEY = ""

    config = {
        'neo4j_uri': NEO4J_URI,
        'neo4j_user': NEO4J_USER,
        'neo4j_password': NEO4J_PASSWORD,
        'gemini_api_key': GEMINI_API_KEY
    }

    DATASET_PATH = "images/geo3k/train"

    # Initialize comparison evaluator
    comp_evaluator = ComparisonEvaluator(config, DATASET_PATH)

    try:
        # Run comparison evaluation with random problem selection
        max_problems = 100  # Change to None to evaluate all problems
        random_seed = 42  # Fixed seed for reproducibility, change to None for random each time

        logger.info(f"Starting comparison evaluation on dataset: {DATASET_PATH}")
        if max_problems:
            logger.info(f"Randomly selecting {max_problems} problems for testing")
        if random_seed:
            logger.info(f"Using random seed: {random_seed} for reproducible problem selection")

        # Run comparison
        comparison_results = comp_evaluator.run_comparison(max_problems=max_problems, random_seed=random_seed)

        # Print summary
        comp_evaluator.print_comparison_summary(comparison_results)

        # Save results
        saved_files = comp_evaluator.save_comparison_results(comparison_results, random_seed=random_seed)

        print("\n" + "="*80)
        print("COMPARISON RESULTS SAVED TO:")
        print("="*80)
        print(f"JSON:   {saved_files['json_path']}")
        print(f"Report: {saved_files['report_path']}")
        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"Error during comparison evaluation: {e}")
        raise


def main():
    """Main evaluation function"""

    # Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = ""
    GEMINI_API_KEY = ""

    config = {
        'neo4j_uri': NEO4J_URI,
        'neo4j_user': NEO4J_USER,
        'neo4j_password': NEO4J_PASSWORD,
        'gemini_api_key': GEMINI_API_KEY
    }

    # Dataset path - update this to your dataset location
    DATASET_PATH = "images/geo3k/train"

    # Choose evaluation mode
    print("\n" + "="*80)
    print("EVALUATION MODE SELECTION")
    print("="*80)
    print("1. Knowledge Graph Only (default)")
    print("2. Knowledge Graph + Image")
    print("3. Comparison (both modes)")
    print("="*80)

    mode = input("Select mode (1/2/3) [default=1]: ").strip() or "1"

    if mode == "3":
        # Run comparison evaluation
        main_comparison()
    else:
        # Run single mode evaluation
        include_image = (mode == "2")
        mode_name = "Knowledge Graph + Image" if include_image else "Knowledge Graph Only"

        logger.info(f"Running evaluation in mode: {mode_name}")

        # Initialize evaluator
        evaluator = GeometricSolverEvaluator(config, DATASET_PATH, include_image=include_image)

        try:
            # Run evaluation with random problem selection
            max_problems = 50  # Change to None to evaluate all problems
            random_seed = 42  # Fixed seed for reproducibility, change to None for random each time

            logger.info(f"Starting evaluation on dataset: {DATASET_PATH}")
            if max_problems:
                logger.info(f"Randomly selecting {max_problems} problems for testing")
            if random_seed:
                logger.info(f"Using random seed: {random_seed} for reproducible problem selection")

            # Evaluate all problems
            metrics = evaluator.evaluate_all(max_problems=max_problems, random_seed=random_seed)

            # Print summary
            evaluator.print_summary()

            # Save results
            saved_files = evaluator.save_results()

            print("\n" + "="*80)
            print("RESULTS SAVED TO:")
            print("="*80)
            print(f"JSON:   {saved_files['json_path']}")
            print(f"CSV:    {saved_files['csv_path']}")
            print(f"Report: {saved_files['report_path']}")
            print("="*80 + "\n")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
        finally:
            evaluator.close()


if __name__ == "__main__":
    # Run main() for interactive mode selection
    # Or uncomment below to run comparison directly
    # main_comparison()
    main()
