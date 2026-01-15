"""
K Selection Strategy Module
Determines adaptive K values for imbalanced queries
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class KStrategy(Enum):
    """K value selection strategies"""
    PERCENTILE = "percentile"  # Percentage of n_pos
    FIXED_CAPPED = "fixed_capped"  # Standard cutoffs capped by n_pos
    ADAPTIVE = "adaptive"  # Stratified adaptive approach (RECOMMENDED)
    CUSTOM = "custom"  # User-defined K values


@dataclass
class KConfig:
    """Configuration for K value selection"""
    strategy: KStrategy
    min_k: int = 1
    max_k: int = 100
    percentiles: List[float] = None  # For PERCENTILE strategy
    fixed_values: List[int] = None  # For FIXED_CAPPED strategy
    custom_k_func: callable = None  # For CUSTOM strategy


class KSelector:
    """
    Determines appropriate K values for each query based on its characteristics
    
    Handles query-specific K selection to account for varying relevance counts
    """
    
    DEFAULT_PERCENTILES = [0.1, 0.25, 0.5, 0.75, 1.0]
    DEFAULT_FIXED_K = [5, 10, 20, 50, 100]
    
    def __init__(self, config: KConfig = None):
        """
        Initialize K selector with configuration
        
        Args:
            config: KConfig object, defaults to ADAPTIVE strategy
        """
        if config is None:
            config = KConfig(strategy=KStrategy.ADAPTIVE)
        
        self.config = config
    
    def get_k_values(self, n_pos: int, query_id: str = None) -> List[int]:
        """
        Get K values for a query based on its positive count
        
        Args:
            n_pos: Number of positive (relevant) documents
            query_id: Optional query identifier for logging
            
        Returns:
            List of K values sorted in ascending order
        """
        if n_pos <= 0:
            raise ValueError(f"n_pos must be positive, got {n_pos}")
        
        if self.config.strategy == KStrategy.PERCENTILE:
            k_values = self._percentile_k(n_pos)
        elif self.config.strategy == KStrategy.FIXED_CAPPED:
            k_values = self._fixed_capped_k(n_pos)
        elif self.config.strategy == KStrategy.ADAPTIVE:
            k_values = self._adaptive_k(n_pos)
        elif self.config.strategy == KStrategy.CUSTOM:
            if self.config.custom_k_func is None:
                raise ValueError("CUSTOM strategy requires custom_k_func")
            k_values = self.config.custom_k_func(n_pos)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        
        # Apply min/max constraints
        k_values = [k for k in k_values if self.config.min_k <= k <= self.config.max_k]
        
        # Ensure all K values are <= n_pos (can't retrieve more than exist)
        k_values = [min(k, n_pos) for k in k_values]
        
        # Remove duplicates and sort
        k_values = sorted(list(set(k_values)))
        
        # Always include n_pos as final K (100% recall possible)
        if n_pos not in k_values:
            k_values.append(n_pos)
            k_values.sort()
        
        return k_values
    
    def _percentile_k(self, n_pos: int) -> List[int]:
        """
        Percentile-based K selection
        
        K values are percentages of n_pos
        """
        percentiles = self.config.percentiles or self.DEFAULT_PERCENTILES
        k_values = [max(1, int(np.ceil(n_pos * p))) for p in percentiles]
        return k_values
    
    def _fixed_capped_k(self, n_pos: int) -> List[int]:
        """
        Fixed K values capped by n_pos
        
        Uses standard IR cutoffs but caps at n_pos
        """
        fixed_k = self.config.fixed_values or self.DEFAULT_FIXED_K
        k_values = [min(k, n_pos) for k in fixed_k if k >= 1]
        return k_values
    
    def _adaptive_k(self, n_pos: int) -> List[int]:
        """
        Adaptive K selection based on query stratum (RECOMMENDED)
        
        Different K ranges for different query sizes:
        - Low positives (< 10): [1, 3, n_pos]
        - Medium positives (10-50): [5, 10, 20, n_pos]
        - High positives (> 50): [10, 20, 50, n_pos]
        """
        if n_pos < 10:
            # Low stratum
            k_values = [1, 3]
        elif n_pos <= 50:
            # Medium stratum
            k_values = [5, 10, 20]
        else:
            # High stratum
            k_values = [10, 20, 50]
        
        # Filter out K values > n_pos
        k_values = [k for k in k_values if k <= n_pos]
        
        return k_values
    
    def get_all_query_k_values(self, query_stats_dict: Dict[str, int]) -> Dict[str, List[int]]:
        """
        Get K values for multiple queries
        
        Args:
            query_stats_dict: Dictionary mapping query_id to n_pos
            
        Returns:
            Dictionary mapping query_id to list of K values
        """
        return {
            query_id: self.get_k_values(n_pos, query_id)
            for query_id, n_pos in query_stats_dict.items()
        }
    
    def get_k_summary(self, query_stats_dict: Dict[str, int]) -> Dict:
        """
        Get summary statistics about K value selection
        
        Args:
            query_stats_dict: Dictionary mapping query_id to n_pos
            
        Returns:
            Dictionary with summary statistics
        """
        all_k_values = self.get_all_query_k_values(query_stats_dict)
        
        k_counts = [len(k_list) for k_list in all_k_values.values()]
        all_k = [k for k_list in all_k_values.values() for k in k_list]
        
        return {
            'total_queries': len(query_stats_dict),
            'avg_k_per_query': np.mean(k_counts),
            'min_k_per_query': min(k_counts),
            'max_k_per_query': max(k_counts),
            'unique_k_values': sorted(list(set(all_k))),
            'total_evaluations': len(all_k)
        }


def create_default_selector() -> KSelector:
    """Create selector with recommended ADAPTIVE strategy"""
    return KSelector(KConfig(strategy=KStrategy.ADAPTIVE))


def create_percentile_selector(percentiles: List[float] = None) -> KSelector:
    """Create selector with PERCENTILE strategy"""
    return KSelector(KConfig(
        strategy=KStrategy.PERCENTILE,
        percentiles=percentiles or [0.1, 0.25, 0.5, 0.75, 1.0]
    ))


def create_fixed_selector(k_values: List[int] = None) -> KSelector:
    """Create selector with FIXED_CAPPED strategy"""
    return KSelector(KConfig(
        strategy=KStrategy.FIXED_CAPPED,
        fixed_values=k_values or [5, 10, 20, 50, 100]
    ))


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("K SELECTION EXAMPLES")
    print("=" * 80)
    
    # Test queries with different characteristics
    test_queries = {
        '06-1P-009': 3,      # Low
        '06-3P-006': 10,     # Low boundary
        '05-3P-002': 28,     # Medium
        '06-1P-004B': 45,    # Medium
        '06-1P-008F': 120,   # High
    }
    
    # Test different strategies
    strategies = [
        ("ADAPTIVE (Recommended)", create_default_selector()),
        ("PERCENTILE", create_percentile_selector()),
        ("FIXED_CAPPED", create_fixed_selector()),
    ]
    
    for strategy_name, selector in strategies:
        print(f"\n{strategy_name} Strategy:")
        print("-" * 80)
        
        for query_id, n_pos in test_queries.items():
            k_values = selector.get_k_values(n_pos, query_id)
            print(f"  {query_id:15s} (n_pos={n_pos:3d}): K = {k_values}")
        
        # Summary
        summary = selector.get_k_summary(test_queries)
        print(f"\nSummary:")
        print(f"  Avg K per query: {summary['avg_k_per_query']:.2f}")
        print(f"  Total evaluations: {summary['total_evaluations']}")
        print(f"  Unique K values: {summary['unique_k_values']}")
