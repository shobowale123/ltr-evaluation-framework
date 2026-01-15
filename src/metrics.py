"""
Metrics Evaluation Module
Computes Recall@K and Precision@K for LTR systems
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class QueryPredictions:
    """Model predictions for a single query"""
    query_id: str
    ranked_doc_ids: List[str]  # Documents ranked by model score (highest first)
    scores: Optional[List[float]] = None  # Optional prediction scores
    
    def __post_init__(self):
        if len(self.ranked_doc_ids) == 0:
            raise ValueError(f"Empty ranking for query {self.query_id}")


@dataclass
class QueryGroundTruth:
    """Ground truth relevance labels for a single query"""
    query_id: str
    relevant_doc_ids: Set[str]  # Set of relevant document IDs
    
    def __post_init__(self):
        if not isinstance(self.relevant_doc_ids, set):
            self.relevant_doc_ids = set(self.relevant_doc_ids)
    
    @property
    def n_relevant(self) -> int:
        return len(self.relevant_doc_ids)


@dataclass
class MetricResult:
    """Result for a single metric at a single K value"""
    query_id: str
    k: int
    metric_name: str
    value: float
    n_relevant_retrieved: int = 0
    n_relevant_total: int = 0
    
    def __repr__(self):
        return (f"{self.metric_name}@{self.k} for {self.query_id}: "
                f"{self.value:.4f} ({self.n_relevant_retrieved}/{self.n_relevant_total})")


@dataclass
class QueryEvaluationResult:
    """Complete evaluation results for a single query"""
    query_id: str
    k_values: List[int]
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    n_relevant: int = 0
    n_retrieved: int = 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy analysis"""
        records = []
        for k in self.k_values:
            records.append({
                'query_id': self.query_id,
                'k': k,
                'recall': self.recall_at_k.get(k, np.nan),
                'precision': self.precision_at_k.get(k, np.nan),
                'n_relevant': self.n_relevant
            })
        return pd.DataFrame(records)


class RankingMetricsEvaluator:
    """
    Computes Recall@K and Precision@K for LTR systems
    
    Handles:
    - Query-specific K values
    - Efficient computation across multiple queries
    - Proper handling of imbalanced data
    """
    
    def __init__(self):
        self.results: Dict[str, QueryEvaluationResult] = {}
    
    def evaluate_query(self,
                      predictions: QueryPredictions,
                      ground_truth: QueryGroundTruth,
                      k_values: List[int]) -> QueryEvaluationResult:
        """
        Evaluate a single query
        
        Args:
            predictions: Model's ranked list for the query
            ground_truth: Relevance labels for the query
            k_values: List of K cutoffs to evaluate
            
        Returns:
            QueryEvaluationResult with metrics at all K values
        """
        if predictions.query_id != ground_truth.query_id:
            raise ValueError("Query ID mismatch between predictions and ground truth")
        
        query_id = predictions.query_id
        ranked_docs = predictions.ranked_doc_ids
        relevant_docs = ground_truth.relevant_doc_ids
        n_relevant = len(relevant_docs)
        
        result = QueryEvaluationResult(
            query_id=query_id,
            k_values=sorted(k_values),
            n_relevant=n_relevant,
            n_retrieved=len(ranked_docs)
        )
        
        # Compute metrics for each K
        for k in k_values:
            if k <= 0:
                raise ValueError(f"K must be positive, got {k}")
            
            # Get top-K predictions
            top_k_docs = ranked_docs[:k]
            
            # Count relevant documents in top-K
            n_relevant_in_k = len(set(top_k_docs) & relevant_docs)
            
            # Recall@K: fraction of relevant docs retrieved
            # When K >= n_relevant, perfect recall is 1.0
            recall = n_relevant_in_k / min(k, n_relevant) if n_relevant > 0 else 0.0
            
            # Precision@K: fraction of retrieved docs that are relevant
            precision = n_relevant_in_k / k if k > 0 else 0.0
            
            result.recall_at_k[k] = recall
            result.precision_at_k[k] = precision
        
        self.results[query_id] = result
        return result
    
    def evaluate_batch(self,
                      predictions_dict: Dict[str, QueryPredictions],
                      ground_truth_dict: Dict[str, QueryGroundTruth],
                      k_values_dict: Dict[str, List[int]]) -> Dict[str, QueryEvaluationResult]:
        """
        Evaluate multiple queries
        
        Args:
            predictions_dict: Dictionary mapping query_id to predictions
            ground_truth_dict: Dictionary mapping query_id to ground truth
            k_values_dict: Dictionary mapping query_id to K values
            
        Returns:
            Dictionary mapping query_id to evaluation results
        """
        results = {}
        
        for query_id in predictions_dict.keys():
            if query_id not in ground_truth_dict:
                raise ValueError(f"Missing ground truth for query {query_id}")
            if query_id not in k_values_dict:
                raise ValueError(f"Missing K values for query {query_id}")
            
            result = self.evaluate_query(
                predictions_dict[query_id],
                ground_truth_dict[query_id],
                k_values_dict[query_id]
            )
            results[query_id] = result
        
        self.results.update(results)
        return results
    
    def get_all_results_dataframe(self) -> pd.DataFrame:
        """
        Get all results as a single DataFrame
        
        Returns:
            DataFrame with columns [query_id, k, recall, precision, n_relevant]
        """
        if not self.results:
            return pd.DataFrame()
        
        dfs = [result.to_dataframe() for result in self.results.values()]
        return pd.concat(dfs, ignore_index=True)
    
    def compute_macro_average(self, metric: str = 'recall') -> Dict[int, float]:
        """
        Compute macro-average (equal weight per query)
        
        Args:
            metric: 'recall' or 'precision'
            
        Returns:
            Dictionary mapping K to macro-averaged metric
        """
        if not self.results:
            return {}
        
        # Get all unique K values
        all_k = set()
        for result in self.results.values():
            all_k.update(result.k_values)
        
        macro_avg = {}
        for k in sorted(all_k):
            values = []
            for result in self.results.values():
                if metric == 'recall' and k in result.recall_at_k:
                    values.append(result.recall_at_k[k])
                elif metric == 'precision' and k in result.precision_at_k:
                    values.append(result.precision_at_k[k])
            
            if values:
                macro_avg[k] = np.mean(values)
        
        return macro_avg
    
    def compute_weighted_average(self, metric: str = 'recall') -> Dict[int, float]:
        """
        Compute weighted average (weighted by n_relevant)
        
        Args:
            metric: 'recall' or 'precision'
            
        Returns:
            Dictionary mapping K to weighted-averaged metric
        """
        if not self.results:
            return {}
        
        # Get all unique K values
        all_k = set()
        for result in self.results.values():
            all_k.update(result.k_values)
        
        weighted_avg = {}
        for k in sorted(all_k):
            weighted_sum = 0.0
            total_weight = 0.0
            
            for result in self.results.values():
                weight = result.n_relevant
                if metric == 'recall' and k in result.recall_at_k:
                    weighted_sum += result.recall_at_k[k] * weight
                    total_weight += weight
                elif metric == 'precision' and k in result.precision_at_k:
                    weighted_sum += result.precision_at_k[k] * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_avg[k] = weighted_sum / total_weight
        
        return weighted_avg
    
    def compute_stratified_metrics(self, 
                                   stratum_dict: Dict[str, List[str]],
                                   metric: str = 'recall') -> Dict[str, Dict[int, float]]:
        """
        Compute macro-average per stratum
        
        Args:
            stratum_dict: Dictionary mapping stratum_name to list of query_ids
            metric: 'recall' or 'precision'
            
        Returns:
            Nested dictionary: stratum_name -> K -> metric value
        """
        stratified = {}
        
        for stratum_name, query_ids in stratum_dict.items():
            # Filter results for this stratum
            stratum_results = {
                qid: result for qid, result in self.results.items()
                if qid in query_ids
            }
            
            if not stratum_results:
                continue
            
            # Get all K values in this stratum
            all_k = set()
            for result in stratum_results.values():
                all_k.update(result.k_values)
            
            # Compute macro-average for stratum
            stratum_metrics = {}
            for k in sorted(all_k):
                values = []
                for result in stratum_results.values():
                    if metric == 'recall' and k in result.recall_at_k:
                        values.append(result.recall_at_k[k])
                    elif metric == 'precision' and k in result.precision_at_k:
                        values.append(result.precision_at_k[k])
                
                if values:
                    stratum_metrics[k] = np.mean(values)
            
            stratified[stratum_name] = stratum_metrics
        
        return stratified


def compute_recall_at_k(ranked_docs: List[str],
                       relevant_docs: Set[str],
                       k: int) -> float:
    """
    Compute Recall@K for a single query
    
    Args:
        ranked_docs: List of document IDs ranked by model
        relevant_docs: Set of relevant document IDs
        k: Cutoff position
        
    Returns:
        Recall@K value
    """
    if k <= 0:
        raise ValueError(f"K must be positive, got {k}")
    
    if len(relevant_docs) == 0:
        return 0.0
    
    top_k = ranked_docs[:k]
    n_relevant_in_k = len(set(top_k) & relevant_docs)
    
    # Recall denominator is min(k, n_relevant)
    return n_relevant_in_k / min(k, len(relevant_docs))


def compute_precision_at_k(ranked_docs: List[str],
                          relevant_docs: Set[str],
                          k: int) -> float:
    """
    Compute Precision@K for a single query
    
    Args:
        ranked_docs: List of document IDs ranked by model
        relevant_docs: Set of relevant document IDs
        k: Cutoff position
        
    Returns:
        Precision@K value
    """
    if k <= 0:
        raise ValueError(f"K must be positive, got {k}")
    
    top_k = ranked_docs[:k]
    n_relevant_in_k = len(set(top_k) & relevant_docs)
    
    return n_relevant_in_k / k


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("METRICS EVALUATION EXAMPLE")
    print("=" * 80)
    
    # Simulate predictions and ground truth
    predictions = QueryPredictions(
        query_id="Q1",
        ranked_doc_ids=["D1", "D5", "D2", "D8", "D3", "D9", "D4", "D6", "D7", "D10"]
    )
    
    ground_truth = QueryGroundTruth(
        query_id="Q1",
        relevant_doc_ids={"D1", "D2", "D3", "D4"}  # 4 relevant docs
    )
    
    evaluator = RankingMetricsEvaluator()
    result = evaluator.evaluate_query(predictions, ground_truth, k_values=[1, 3, 5, 10])
    
    print(f"\nQuery: {result.query_id}")
    print(f"Relevant documents: {ground_truth.n_relevant}")
    print("\nMetrics:")
    print("-" * 80)
    for k in result.k_values:
        recall = result.recall_at_k[k]
        precision = result.precision_at_k[k]
        print(f"  K={k:2d}: Recall={recall:.4f}, Precision={precision:.4f}")
