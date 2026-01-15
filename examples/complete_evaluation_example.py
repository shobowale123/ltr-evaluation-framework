"""
Complete Example: Evaluating LTR System with Provided Data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from query_statistics import load_query_data_from_text
from pipeline import LTREvaluationPipeline
from metrics import QueryPredictions, QueryGroundTruth


def load_sample_data():
    """Load the provided query statistics"""
    sample_data = """04-CWP-005	78	3243
05-3P-001C	66	1859
05-3P-002	28	1331
05-CWP-011	15	1219
06-1P-004B	45	1648
06-1P-008F	120	4966
06-1P-009	3	1036
06-1P-010	3	549
06-1P-014D	120	2969
06-1P-015	3	510
06-3P-001	78	2115
06-3P-006	10	1706
06-3P-008	28	1289
06-5P-007	28	1910
07-1P-004	10	641
07-1P-006	55	1776
07-1P-007	15	1066
07-1P-010	15	1060
07-2P-005	10	1086
07-2P-006	36	2318
07-2P-007	36	1421
07-2P-009	6	590
07-3P-001	28	2750
07-3P-002	3	837"""
    
    return load_query_data_from_text(sample_data)


def simulate_model_predictions(query_data: pd.DataFrame, 
                               quality: str = 'good') -> dict:
    """
    Simulate LTR model predictions
    
    In real usage, this would load actual model predictions from files
    
    Args:
        query_data: DataFrame with query statistics
        quality: 'good', 'medium', or 'poor' for simulation
        
    Returns:
        Dictionary mapping query_id to QueryPredictions
    """
    np.random.seed(42)
    
    predictions = {}
    
    for _, row in query_data.iterrows():
        query_id = row['query_id']
        n_pos = row['n_pos']
        n_neg = row['n_neg']
        n_total = n_pos + n_neg
        
        # Generate document IDs
        relevant_docs = [f"{query_id}_REL_{i}" for i in range(n_pos)]
        non_relevant_docs = [f"{query_id}_NREL_{i}" for i in range(n_neg)]
        all_docs = relevant_docs + non_relevant_docs
        
        # Simulate ranking based on quality
        if quality == 'good':
            # Good model: mostly relevant docs at top
            scores = []
            # Relevant docs get higher scores
            for doc in relevant_docs:
                scores.append(np.random.beta(8, 2))  # High scores
            # Non-relevant docs get lower scores
            for doc in non_relevant_docs:
                scores.append(np.random.beta(2, 8))  # Low scores
            
        elif quality == 'medium':
            # Medium model: some relevant docs at top
            scores = []
            for doc in relevant_docs:
                scores.append(np.random.beta(5, 3))  # Medium-high scores
            for doc in non_relevant_docs:
                scores.append(np.random.beta(3, 5))  # Medium-low scores
                
        else:  # poor
            # Poor model: random ranking
            scores = [np.random.random() for _ in all_docs]
        
        # Sort documents by score (descending)
        doc_score_pairs = list(zip(all_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        ranked_docs = [doc for doc, _ in doc_score_pairs]
        ranked_scores = [score for _, score in doc_score_pairs]
        
        predictions[query_id] = QueryPredictions(
            query_id=query_id,
            ranked_doc_ids=ranked_docs,
            scores=ranked_scores
        )
    
    return predictions


def create_ground_truth(query_data: pd.DataFrame) -> dict:
    """
    Create ground truth labels
    
    In real usage, this would load actual ground truth from files
    
    Args:
        query_data: DataFrame with query statistics
        
    Returns:
        Dictionary mapping query_id to QueryGroundTruth
    """
    ground_truth = {}
    
    for _, row in query_data.iterrows():
        query_id = row['query_id']
        n_pos = row['n_pos']
        
        # Generate relevant document IDs (must match prediction IDs)
        relevant_docs = {f"{query_id}_REL_{i}" for i in range(n_pos)}
        
        ground_truth[query_id] = QueryGroundTruth(
            query_id=query_id,
            relevant_doc_ids=relevant_docs
        )
    
    return ground_truth


def main():
    """Run complete evaluation example"""
    
    print("=" * 80)
    print("LTR EVALUATION FRAMEWORK - Complete Example")
    print("=" * 80)
    
    # 1. Load query statistics
    print("\n[Step 1] Loading query statistics...")
    query_data = load_sample_data()
    print(f"✓ Loaded {len(query_data)} queries")
    print(f"\nQuery statistics overview:")
    print(query_data.describe())
    
    # 2. Initialize pipeline
    print("\n[Step 2] Initializing evaluation pipeline...")
    pipeline = LTREvaluationPipeline(
        output_dir="evaluation_results",
        k_strategy="adaptive"  # Recommended for imbalanced data
    )
    
    # 3. Load query statistics into pipeline
    print("\n[Step 3] Analyzing query characteristics...")
    query_analyzer = pipeline.load_query_statistics(query_data)
    print("\n" + query_analyzer.get_summary_report())
    
    # 4. Create/Load predictions and ground truth
    print("\n[Step 4] Loading model predictions and ground truth...")
    predictions = simulate_model_predictions(query_data, quality='good')
    ground_truth = create_ground_truth(query_data)
    print(f"✓ Loaded predictions for {len(predictions)} queries")
    print(f"✓ Loaded ground truth for {len(ground_truth)} queries")
    
    # 5. Run evaluation
    print("\n[Step 5] Running evaluation...")
    results = pipeline.evaluate_predictions(predictions, ground_truth)
    
    # 6. Compute aggregate metrics
    print("\n[Step 6] Computing aggregate metrics...")
    aggregates = pipeline.compute_aggregate_metrics()
    
    print("\n" + "=" * 80)
    print("MACRO-AVERAGED METRICS (Equal weight per query)")
    print("=" * 80)
    print("\nRecall@K:")
    for k, value in sorted(aggregates['recall']['macro'].items()):
        print(f"  K={k:3d}: {value:.4f}")
    
    print("\nPrecision@K:")
    for k, value in sorted(aggregates['precision']['macro'].items()):
        print(f"  K={k:3d}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("WEIGHTED-AVERAGED METRICS (Weighted by relevance count)")
    print("=" * 80)
    print("\nRecall@K:")
    for k, value in sorted(aggregates['recall']['weighted'].items()):
        print(f"  K={k:3d}: {value:.4f}")
    
    print("\nPrecision@K:")
    for k, value in sorted(aggregates['precision']['weighted'].items()):
        print(f"  K={k:3d}: {value:.4f}")
    
    # 7. Stratified analysis
    print("\n[Step 7] Computing stratified metrics...")
    stratified = pipeline.compute_stratified_metrics()
    
    print("\n" + "=" * 80)
    print("STRATIFIED ANALYSIS - Recall@K")
    print("=" * 80)
    for stratum, metrics in stratified['recall'].items():
        print(f"\n{stratum}:")
        for k, value in sorted(metrics.items()):
            print(f"  K={k:3d}: {value:.4f}")
    
    # 8. Confidence intervals
    print("\n[Step 8] Computing confidence intervals...")
    cis = pipeline.compute_confidence_intervals(n_iterations=1000)
    
    print("\n" + "=" * 80)
    print("95% CONFIDENCE INTERVALS - Recall@K")
    print("=" * 80)
    for ci in cis['recall']:
        print(f"K={ci.k:3d}: {ci.mean:.4f} [{ci.lower:.4f}, {ci.upper:.4f}] "
              f"(width: {ci.width:.4f})")
    
    # 9. Query difficulty analysis
    print("\n[Step 9] Analyzing query difficulty correlation...")
    difficulty_corr = pipeline.analyze_query_difficulty()
    
    print("\n" + "=" * 80)
    print("DIFFICULTY CORRELATION (Spearman ρ)")
    print("=" * 80)
    print(difficulty_corr.to_string(index=False))
    
    # 10. Generate full report
    print("\n[Step 10] Generating comprehensive report...")
    report_path = pipeline.generate_full_report("ltr_evaluation_report")
    
    # Additional: Compare with baseline
    print("\n[Step 11 - Optional] Comparing with baseline system...")
    baseline_predictions = simulate_model_predictions(query_data, quality='medium')
    
    comparison = pipeline.compare_systems(
        baseline_predictions,
        ground_truth,
        system1_name="LTR Model",
        system2_name="Baseline"
    )
    
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS (LTR Model vs Baseline)")
    print("=" * 80)
    for k, test_result in comparison['statistical_tests'].items():
        sig_marker = "***" if test_result.is_significant else "n.s."
        print(f"K={k:3d}: p={test_result.p_value:.4f} {sig_marker}, "
              f"Cohen's d={test_result.effect_size:.3f}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {pipeline.output_dir}")
    print(f"HTML Report: {report_path}")
    print("\nGenerated files:")
    print("  - detailed_results.csv: Per-query, per-K metrics")
    print("  - summary_metrics.json: Aggregate metrics")
    print("  - recall_vs_k.png: Recall@K visualization")
    print("  - precision_vs_k.png: Precision@K visualization")
    print("  - recall_heatmap.png: Per-query performance heatmap")
    print("  - difficulty_correlation_kX.png: Difficulty analysis")
    print("  - system_comparison.png: Model vs baseline comparison")
    print("  - ltr_evaluation_report.html: Complete HTML report")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FOR YOUR DATA:")
    print("=" * 80)
    print("\n1. QUERY IMBALANCE:")
    print("   - Relevant docs per query: 3 to 120 (highly variable)")
    print("   - Used ADAPTIVE K strategy to handle this")
    print("   - Low queries (≤10 rel): K=[1,3,n_pos]")
    print("   - Medium queries (11-50 rel): K=[5,10,20,n_pos]")
    print("   - High queries (>50 rel): K=[10,20,50,n_pos]")
    
    print("\n2. CLASS IMBALANCE:")
    print("   - Imbalance ratio ranges from 1:41 to 1:170")
    print("   - Used query-specific K to ensure fair evaluation")
    print("   - Computed both macro and weighted averages")
    
    print("\n3. RECOMMENDED ACTIONS:")
    print("   - Review stratified metrics to identify query type weaknesses")
    print("   - Check difficulty correlation - negative means model struggles with imbalance")
    print("   - Use macro-average for equal query importance")
    print("   - Use weighted-average when large queries matter more")
    print("   - Bootstrap CIs quantify uncertainty in estimates")
    
    print("\n4. NEXT STEPS:")
    print("   - Replace simulated predictions with actual model outputs")
    print("   - Load ground truth from your LTR dataset")
    print("   - Compare multiple model variants")
    print("   - Tune model focusing on low-performing strata")
    

if __name__ == "__main__":
    main()
