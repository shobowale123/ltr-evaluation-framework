"""
End-to-End LTR Evaluation Pipeline
Orchestrates complete evaluation workflow
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from query_statistics import QueryStatisticsAnalyzer, load_query_data_from_text
from k_selector import KSelector, KConfig, KStrategy, create_default_selector
from metrics import (RankingMetricsEvaluator, QueryPredictions, 
                     QueryGroundTruth, QueryEvaluationResult)
from statistical_tests import BootstrapCI, PermutationTest, VarianceAnalyzer
from reporting import EvaluationReporter


class LTREvaluationPipeline:
    """
    Complete end-to-end evaluation pipeline for LTR systems
    
    Usage:
        1. Load query statistics
        2. Load model predictions and ground truth
        3. Configure K selection strategy
        4. Run evaluation
        5. Generate reports and visualizations
    """
    
    def __init__(self, 
                 output_dir: str = "evaluation_results",
                 k_strategy: str = "adaptive",
                 random_seed: int = 42):
        """
        Initialize evaluation pipeline
        
        Args:
            output_dir: Directory for saving results
            k_strategy: Strategy for K value selection ('adaptive', 'percentile', 'fixed_capped')
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Convert string to enum
        if isinstance(k_strategy, str):
            strategy_map = {
                'adaptive': KStrategy.ADAPTIVE,
                'percentile': KStrategy.PERCENTILE,
                'fixed_capped': KStrategy.FIXED_CAPPED,
                'custom': KStrategy.CUSTOM
            }
            k_strategy = strategy_map.get(k_strategy.lower(), KStrategy.ADAPTIVE)
        
        # Initialize components
        self.k_selector = KSelector(KConfig(strategy=k_strategy))
        self.evaluator = RankingMetricsEvaluator()
        self.reporter = EvaluationReporter(output_dir=str(self.output_dir))
        
        # Storage
        self.query_analyzer = None
        self.results = {}
        self.k_values_dict = {}
        
    def load_query_statistics(self, 
                             query_data: pd.DataFrame) -> QueryStatisticsAnalyzer:
        """
        Load and analyze query statistics
        
        Args:
            query_data: DataFrame with columns [query_id, n_pos, n_neg]
            
        Returns:
            QueryStatisticsAnalyzer object
        """
        self.query_analyzer = QueryStatisticsAnalyzer(query_data)
        
        # Determine K values for each query
        self.k_values_dict = {}
        for query_id in query_data['query_id']:
            stats = self.query_analyzer.get_query_stats(query_id)
            self.k_values_dict[query_id] = self.k_selector.get_k_values(stats.n_pos)
        
        print(f"✓ Loaded statistics for {len(query_data)} queries")
        print(f"✓ K selection strategy: {self.k_selector.config.strategy.value}")
        
        return self.query_analyzer
    
    def evaluate_predictions(self,
                           predictions_dict: Dict[str, QueryPredictions],
                           ground_truth_dict: Dict[str, QueryGroundTruth]) -> Dict[str, QueryEvaluationResult]:
        """
        Evaluate model predictions
        
        Args:
            predictions_dict: Dictionary mapping query_id to predictions
            ground_truth_dict: Dictionary mapping query_id to ground truth
            
        Returns:
            Dictionary mapping query_id to evaluation results
        """
        if self.query_analyzer is None:
            raise ValueError("Must load query statistics first")
        
        # Validate queries
        pred_queries = set(predictions_dict.keys())
        gt_queries = set(ground_truth_dict.keys())
        stats_queries = set(self.k_values_dict.keys())
        
        common_queries = pred_queries & gt_queries & stats_queries
        
        if len(common_queries) == 0:
            raise ValueError("No common queries found across all inputs")
        
        if len(common_queries) < len(pred_queries):
            missing = pred_queries - common_queries
            print(f"⚠ Warning: {len(missing)} queries missing ground truth or statistics")
        
        # Filter to common queries
        predictions_dict = {q: predictions_dict[q] for q in common_queries}
        ground_truth_dict = {q: ground_truth_dict[q] for q in common_queries}
        k_values_dict = {q: self.k_values_dict[q] for q in common_queries}
        
        # Run evaluation
        self.results = self.evaluator.evaluate_batch(
            predictions_dict,
            ground_truth_dict,
            k_values_dict
        )
        
        print(f"✓ Evaluated {len(self.results)} queries")
        
        return self.results
    
    def compute_aggregate_metrics(self) -> Dict:
        """
        Compute aggregate metrics across all queries
        
        Returns:
            Dictionary with macro and weighted averages
        """
        if not self.results:
            raise ValueError("Must run evaluation first")
        
        aggregates = {
            'recall': {
                'macro': self.evaluator.compute_macro_average('recall'),
                'weighted': self.evaluator.compute_weighted_average('recall')
            },
            'precision': {
                'macro': self.evaluator.compute_macro_average('precision'),
                'weighted': self.evaluator.compute_weighted_average('precision')
            }
        }
        
        print("✓ Computed aggregate metrics")
        
        return aggregates
    
    def compute_stratified_metrics(self) -> Dict:
        """
        Compute metrics stratified by query characteristics
        
        Returns:
            Dictionary with stratified metrics
        """
        if not self.results or self.query_analyzer is None:
            raise ValueError("Must run evaluation and load query statistics first")
        
        # Get stratification
        stratification = self.query_analyzer.stratify_queries()
        stratum_dict = {
            stratum.value: query_ids 
            for stratum, query_ids in stratification.items()
        }
        
        stratified = {
            'recall': self.evaluator.compute_stratified_metrics(stratum_dict, 'recall'),
            'precision': self.evaluator.compute_stratified_metrics(stratum_dict, 'precision')
        }
        
        print("✓ Computed stratified metrics")
        
        return stratified
    
    def compute_confidence_intervals(self, 
                                    n_iterations: int = 1000,
                                    confidence_level: float = 0.95) -> Dict:
        """
        Compute bootstrap confidence intervals
        
        Args:
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level (e.g., 0.95)
            
        Returns:
            Dictionary with confidence intervals
        """
        if not self.results:
            raise ValueError("Must run evaluation first")
        
        results_df = self.evaluator.get_all_results_dataframe()
        bootstrap = BootstrapCI(n_iterations, confidence_level, self.random_seed)
        
        cis = {
            'recall': bootstrap.compute_multiple_ci(results_df, 'recall'),
            'precision': bootstrap.compute_multiple_ci(results_df, 'precision')
        }
        
        print(f"✓ Computed {confidence_level*100:.0f}% confidence intervals")
        
        return cis
    
    def analyze_query_difficulty(self) -> pd.DataFrame:
        """
        Analyze correlation between query difficulty and performance
        
        Returns:
            DataFrame with difficulty analysis
        """
        if not self.results or self.query_analyzer is None:
            raise ValueError("Must run evaluation and load query statistics first")
        
        results_df = self.evaluator.get_all_results_dataframe()
        difficulty_df = self.query_analyzer.get_difficulty_distribution()
        
        # Merge with results
        merged = results_df.merge(difficulty_df[['query_id', 'difficulty_score']], 
                                 on='query_id')
        
        # Compute correlations for each K
        correlations = []
        for k in merged['k'].unique():
            k_data = merged[merged['k'] == k]
            
            recall_corr = k_data['difficulty_score'].corr(k_data['recall'])
            precision_corr = k_data['difficulty_score'].corr(k_data['precision'])
            
            correlations.append({
                'k': k,
                'recall_correlation': recall_corr,
                'precision_correlation': precision_corr
            })
        
        corr_df = pd.DataFrame(correlations)
        
        print("✓ Analyzed query difficulty correlations")
        
        return corr_df
    
    def generate_full_report(self, report_name: str = "ltr_evaluation_report"):
        """
        Generate comprehensive evaluation report
        
        Args:
            report_name: Name for the report
        """
        if not self.results or self.query_analyzer is None:
            raise ValueError("Must run complete evaluation first")
        
        # Get all components
        results_df = self.evaluator.get_all_results_dataframe()
        query_stats_df = self.query_analyzer.data
        aggregates = self.compute_aggregate_metrics()
        stratified = self.compute_stratified_metrics()
        
        # Generate plots
        print("\nGenerating visualizations...")
        
        # 1. Recall vs K
        self.reporter.plot_metrics_vs_k(
            results_df, 'recall',
            save_path=self.output_dir / "recall_vs_k.png"
        )
        
        # 2. Precision vs K
        self.reporter.plot_metrics_vs_k(
            results_df, 'precision',
            save_path=self.output_dir / "precision_vs_k.png"
        )
        
        # 3. Query heatmap
        self.reporter.plot_query_heatmap(
            results_df, 'recall', top_n=20,
            save_path=self.output_dir / "recall_heatmap.png"
        )
        
        # 4. Difficulty correlation
        difficulty_df = self.query_analyzer.get_difficulty_distribution()
        # Get median K for analysis
        median_k = int(np.median(results_df['k'].unique()))
        self.reporter.plot_difficulty_correlation(
            difficulty_df, results_df, 'recall', median_k,
            save_path=self.output_dir / f"difficulty_correlation_k{median_k}.png"
        )
        
        # Save data files
        print("\nSaving data files...")
        self.reporter.save_results_csv(results_df, "detailed_results.csv")
        self.reporter.save_summary_json(
            aggregates['recall']['macro'],
            aggregates['recall']['weighted'],
            "summary_metrics.json"
        )
        
        # Generate HTML report
        print("\nGenerating HTML report...")
        report_path = self.reporter.generate_full_report(
            results_df,
            query_stats_df,
            {'recall': aggregates['recall']['macro'], 
             'precision': aggregates['precision']['macro']},
            {'recall': aggregates['recall']['weighted'],
             'precision': aggregates['precision']['weighted']},
            stratified,
            report_name
        )
        
        print(f"\n{'='*80}")
        print(f"✓ Evaluation complete!")
        print(f"✓ Results saved to: {self.output_dir}")
        print(f"✓ HTML report: {report_path}")
        print(f"{'='*80}\n")
        
        return report_path
    
    def compare_systems(self,
                       system2_predictions: Dict[str, QueryPredictions],
                       ground_truth_dict: Dict[str, QueryGroundTruth],
                       system1_name: str = "System 1",
                       system2_name: str = "System 2") -> Dict:
        """
        Compare two LTR systems with statistical testing
        
        Args:
            system2_predictions: Predictions from second system
            ground_truth_dict: Ground truth labels
            system1_name: Name of first system (current)
            system2_name: Name of second system
            
        Returns:
            Dictionary with comparison results
        """
        if not self.results:
            raise ValueError("Must evaluate first system before comparison")
        
        # Evaluate system 2
        evaluator2 = RankingMetricsEvaluator()
        results2 = evaluator2.evaluate_batch(
            system2_predictions,
            ground_truth_dict,
            self.k_values_dict
        )
        
        # Get common queries
        common_queries = set(self.results.keys()) & set(results2.keys())
        
        # Perform permutation tests
        perm_test = PermutationTest(n_permutations=1000, random_seed=self.random_seed)
        
        test_results = {}
        results_df1 = self.evaluator.get_all_results_dataframe()
        results_df2 = evaluator2.get_all_results_dataframe()
        
        for k in sorted(results_df1['k'].unique()):
            k_data1 = results_df1[results_df1['k'] == k]
            k_data2 = results_df2[results_df2['k'] == k]
            
            metrics1_recall = dict(zip(k_data1['query_id'], k_data1['recall']))
            metrics2_recall = dict(zip(k_data2['query_id'], k_data2['recall']))
            
            test_result = perm_test.test(
                metrics1_recall, metrics2_recall, 'recall', k
            )
            
            test_results[k] = test_result
        
        # Generate comparison plots
        macro1 = self.evaluator.compute_macro_average('recall')
        macro2 = evaluator2.compute_macro_average('recall')
        
        self.reporter.plot_comparison(
            macro1, macro2, system1_name, system2_name, 'recall',
            save_path=self.output_dir / "system_comparison.png"
        )
        
        print(f"✓ Comparison complete: {system1_name} vs {system2_name}")
        
        return {
            'system1_results': self.results,
            'system2_results': results2,
            'statistical_tests': test_results
        }


if __name__ == "__main__":
    print("=" * 80)
    print("LTR EVALUATION PIPELINE - Example Usage")
    print("=" * 80)
    
    # This would be replaced with actual data loading
    print("\nNote: This is a template. See examples/ directory for complete usage.")
    print("\nBasic workflow:")
    print("1. Initialize pipeline")
    print("2. Load query statistics")
    print("3. Load predictions and ground truth")
    print("4. Run evaluation")
    print("5. Generate reports")
