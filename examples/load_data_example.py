"""
Example: Loading Your Real LTR Data
Shows how to load actual model predictions and ground truth from files
"""

import pandas as pd
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from query_statistics import load_query_data_from_text
from pipeline import LTREvaluationPipeline
from metrics import QueryPredictions, QueryGroundTruth


# =============================================================================
# OPTION 1: Load from CSV Files
# =============================================================================

def load_predictions_from_csv(predictions_file: str) -> dict:
    """
    Load model predictions from CSV file
    
    Expected CSV format:
        query_id,doc_id,rank,score
        04-CWP-005,doc123,1,0.95
        04-CWP-005,doc456,2,0.87
        04-CWP-005,doc789,3,0.82
        ...
    
    Or simpler format (if already ranked):
        query_id,doc_id
        04-CWP-005,doc123
        04-CWP-005,doc456
        04-CWP-005,doc789
        ...
    """
    df = pd.read_csv(predictions_file)
    
    predictions = {}
    
    for query_id, group in df.groupby('query_id'):
        # Sort by rank or score if available
        if 'rank' in group.columns:
            group = group.sort_values('rank')
        elif 'score' in group.columns:
            group = group.sort_values('score', ascending=False)
        
        ranked_docs = group['doc_id'].tolist()
        scores = group['score'].tolist() if 'score' in group.columns else None
        
        predictions[query_id] = QueryPredictions(
            query_id=query_id,
            ranked_doc_ids=ranked_docs,
            scores=scores
        )
    
    return predictions


def load_ground_truth_from_csv(ground_truth_file: str) -> dict:
    """
    Load ground truth from CSV file
    
    Expected CSV format:
        query_id,doc_id,relevance
        04-CWP-005,doc123,1
        04-CWP-005,doc789,1
        05-3P-001C,doc456,1
        ...
    
    Or simpler (binary relevance):
        query_id,doc_id
        04-CWP-005,doc123
        04-CWP-005,doc789
        ...
    """
    df = pd.read_csv(ground_truth_file)
    
    ground_truth = {}
    
    for query_id, group in df.groupby('query_id'):
        # If relevance column exists, filter for relevant docs
        if 'relevance' in group.columns:
            relevant_docs = set(group[group['relevance'] > 0]['doc_id'])
        else:
            # All listed docs are relevant
            relevant_docs = set(group['doc_id'])
        
        ground_truth[query_id] = QueryGroundTruth(
            query_id=query_id,
            relevant_doc_ids=relevant_docs
        )
    
    return ground_truth


# =============================================================================
# OPTION 2: Load from JSON Files
# =============================================================================

def load_predictions_from_json(predictions_file: str) -> dict:
    """
    Load predictions from JSON file
    
    Expected JSON format:
    {
        "04-CWP-005": {
            "ranked_docs": ["doc123", "doc456", "doc789", ...],
            "scores": [0.95, 0.87, 0.82, ...]  // optional
        },
        "05-3P-001C": {
            "ranked_docs": [...],
            "scores": [...]
        }
    }
    """
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    predictions = {}
    
    for query_id, query_data in data.items():
        predictions[query_id] = QueryPredictions(
            query_id=query_id,
            ranked_doc_ids=query_data['ranked_docs'],
            scores=query_data.get('scores')
        )
    
    return predictions


def load_ground_truth_from_json(ground_truth_file: str) -> dict:
    """
    Load ground truth from JSON file
    
    Expected JSON format:
    {
        "04-CWP-005": ["doc123", "doc789", ...],
        "05-3P-001C": ["doc456", ...],
        ...
    }
    """
    with open(ground_truth_file, 'r') as f:
        data = json.load(f)
    
    ground_truth = {}
    
    for query_id, relevant_docs in data.items():
        ground_truth[query_id] = QueryGroundTruth(
            query_id=query_id,
            relevant_doc_ids=set(relevant_docs)
        )
    
    return ground_truth


# =============================================================================
# OPTION 3: Load from TREC Format
# =============================================================================

def load_predictions_from_trec(run_file: str) -> dict:
    """
    Load predictions from TREC run file
    
    Expected format (space or tab separated):
        query_id Q0 doc_id rank score run_name
        04-CWP-005 Q0 doc123 1 0.95 my_run
        04-CWP-005 Q0 doc456 2 0.87 my_run
        ...
    """
    df = pd.read_csv(
        run_file,
        sep=r'\s+',
        names=['query_id', 'q0', 'doc_id', 'rank', 'score', 'run_name']
    )
    
    predictions = {}
    
    for query_id, group in df.groupby('query_id'):
        group = group.sort_values('rank')
        
        predictions[query_id] = QueryPredictions(
            query_id=query_id,
            ranked_doc_ids=group['doc_id'].tolist(),
            scores=group['score'].tolist()
        )
    
    return predictions


def load_ground_truth_from_qrels(qrels_file: str) -> dict:
    """
    Load ground truth from TREC qrels file
    
    Expected format (space or tab separated):
        query_id 0 doc_id relevance
        04-CWP-005 0 doc123 1
        04-CWP-005 0 doc789 1
        ...
    """
    df = pd.read_csv(
        qrels_file,
        sep=r'\s+',
        names=['query_id', 'iteration', 'doc_id', 'relevance']
    )
    
    ground_truth = {}
    
    for query_id, group in df.groupby('query_id'):
        # Consider docs with relevance > 0 as relevant
        relevant_docs = set(group[group['relevance'] > 0]['doc_id'])
        
        ground_truth[query_id] = QueryGroundTruth(
            query_id=query_id,
            relevant_doc_ids=relevant_docs
        )
    
    return ground_truth


# =============================================================================
# COMPLETE WORKFLOW EXAMPLE
# =============================================================================

def evaluate_ltr_system_from_files(
    query_stats_file: str,
    predictions_file: str,
    ground_truth_file: str,
    file_format: str = 'csv',
    output_dir: str = 'my_evaluation_results'
):
    """
    Complete evaluation workflow from files
    
    Args:
        query_stats_file: Path to query statistics (CSV or text)
        predictions_file: Path to model predictions
        ground_truth_file: Path to ground truth labels
        file_format: 'csv', 'json', or 'trec'
        output_dir: Directory for results
    """
    
    print("=" * 80)
    print("LTR SYSTEM EVALUATION FROM FILES")
    print("=" * 80)
    
    # 1. Load query statistics
    print("\n[1/5] Loading query statistics...")
    if query_stats_file.endswith('.csv'):
        query_data = pd.read_csv(query_stats_file)
    else:
        with open(query_stats_file, 'r') as f:
            query_data = load_query_data_from_text(f.read())
    
    print(f"✓ Loaded {len(query_data)} queries")
    
    # 2. Load predictions
    print("\n[2/5] Loading model predictions...")
    if file_format == 'csv':
        predictions = load_predictions_from_csv(predictions_file)
    elif file_format == 'json':
        predictions = load_predictions_from_json(predictions_file)
    elif file_format == 'trec':
        predictions = load_predictions_from_trec(predictions_file)
    else:
        raise ValueError(f"Unknown format: {file_format}")
    
    print(f"✓ Loaded predictions for {len(predictions)} queries")
    
    # 3. Load ground truth
    print("\n[3/5] Loading ground truth...")
    if file_format == 'csv':
        ground_truth = load_ground_truth_from_csv(ground_truth_file)
    elif file_format == 'json':
        ground_truth = load_ground_truth_from_json(ground_truth_file)
    elif file_format == 'trec':
        ground_truth = load_ground_truth_from_qrels(ground_truth_file)
    
    print(f"✓ Loaded ground truth for {len(ground_truth)} queries")
    
    # 4. Run evaluation
    print("\n[4/5] Running evaluation...")
    pipeline = LTREvaluationPipeline(
        output_dir=output_dir,
        k_strategy='adaptive'
    )
    
    pipeline.load_query_statistics(query_data)
    pipeline.evaluate_predictions(predictions, ground_truth)
    
    # 5. Generate report
    print("\n[5/5] Generating comprehensive report...")
    report_path = pipeline.generate_full_report()
    
    # Display summary
    aggregates = pipeline.compute_aggregate_metrics()
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print("\nTop-5 Recall@K (Macro-Average):")
    recall_items = sorted(aggregates['recall']['macro'].items())[:5]
    for k, value in recall_items:
        print(f"  K={k:3d}: {value:.4f}")
    
    print("\nTop-5 Precision@K (Macro-Average):")
    precision_items = sorted(aggregates['precision']['macro'].items())[:5]
    for k, value in precision_items:
        print(f"  K={k:3d}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print(f"✓ Complete results saved to: {output_dir}/")
    print(f"✓ Open HTML report: {report_path}")
    print("=" * 80)
    
    return pipeline


# =============================================================================
# EXAMPLE DATA CREATION (for testing)
# =============================================================================

def create_example_csv_files():
    """Create example CSV files for testing"""
    
    # Example predictions
    predictions_data = {
        'query_id': ['Q1'] * 5 + ['Q2'] * 5,
        'doc_id': ['D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D12', 'D13', 'D14'],
        'rank': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'score': [0.9, 0.8, 0.7, 0.6, 0.5, 0.95, 0.85, 0.75, 0.65, 0.55]
    }
    pd.DataFrame(predictions_data).to_csv('example_predictions.csv', index=False)
    
    # Example ground truth
    ground_truth_data = {
        'query_id': ['Q1', 'Q1', 'Q1', 'Q2', 'Q2'],
        'doc_id': ['D1', 'D3', 'D5', 'D10', 'D12'],
        'relevance': [1, 1, 1, 1, 1]
    }
    pd.DataFrame(ground_truth_data).to_csv('example_ground_truth.csv', index=False)
    
    # Example query stats
    query_stats_data = {
        'query_id': ['Q1', 'Q2'],
        'n_pos': [3, 2],
        'n_neg': [10, 8]
    }
    pd.DataFrame(query_stats_data).to_csv('example_query_stats.csv', index=False)
    
    print("Created example CSV files:")
    print("  - example_predictions.csv")
    print("  - example_ground_truth.csv")
    print("  - example_query_stats.csv")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate LTR system from files')
    parser.add_argument('--query-stats', help='Query statistics file')
    parser.add_argument('--predictions', help='Model predictions file')
    parser.add_argument('--ground-truth', help='Ground truth file')
    parser.add_argument('--format', choices=['csv', 'json', 'trec'], default='csv')
    parser.add_argument('--output', default='evaluation_results')
    parser.add_argument('--create-example', action='store_true', 
                       help='Create example CSV files')
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_csv_files()
        print("\nNow run:")
        print("python load_data_example.py --query-stats example_query_stats.csv "
              "--predictions example_predictions.csv "
              "--ground-truth example_ground_truth.csv")
    elif args.query_stats and args.predictions and args.ground_truth:
        evaluate_ltr_system_from_files(
            args.query_stats,
            args.predictions,
            args.ground_truth,
            args.format,
            args.output
        )
    else:
        parser.print_help()
        print("\n" + "=" * 80)
        print("EXAMPLES:")
        print("=" * 80)
        print("\n1. Create example files:")
        print("   python load_data_example.py --create-example")
        print("\n2. Evaluate from CSV:")
        print("   python load_data_example.py \\")
        print("       --query-stats query_stats.csv \\")
        print("       --predictions predictions.csv \\")
        print("       --ground-truth ground_truth.csv \\")
        print("       --format csv")
        print("\n3. Evaluate from TREC format:")
        print("   python load_data_example.py \\")
        print("       --query-stats query_stats.txt \\")
        print("       --predictions run.txt \\")
        print("       --ground-truth qrels.txt \\")
        print("       --format trec")
