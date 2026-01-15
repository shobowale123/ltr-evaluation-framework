"""
Statistical Analysis Module
Confidence intervals, significance testing, and effect sizes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric"""
    metric_name: str
    k: int
    mean: float
    lower: float
    upper: float
    confidence_level: float
    n_samples: int
    
    def __repr__(self):
        return (f"{self.metric_name}@{self.k}: {self.mean:.4f} "
                f"[{self.lower:.4f}, {self.upper:.4f}] "
                f"({self.confidence_level*100:.0f}% CI)")
    
    @property
    def width(self) -> float:
        """Width of confidence interval"""
        return self.upper - self.lower
    
    @property
    def relative_width(self) -> float:
        """Width relative to mean"""
        return self.width / self.mean if self.mean > 0 else float('inf')


@dataclass
class SignificanceTestResult:
    """Result of statistical significance test"""
    test_name: str
    metric_name: str
    k: int
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    effect_size: Optional[float] = None
    
    def __repr__(self):
        sig_str = "SIGNIFICANT" if self.is_significant else "not significant"
        return (f"{self.test_name} for {self.metric_name}@{self.k}: "
                f"p={self.p_value:.4f} ({sig_str} at α={self.alpha})")


class BootstrapCI:
    """
    Bootstrap confidence intervals for ranking metrics
    
    Performs query-level resampling to compute confidence intervals
    """
    
    def __init__(self, 
                 n_iterations: int = 1000,
                 confidence_level: float = 0.95,
                 random_seed: Optional[int] = 42):
        """
        Initialize bootstrap confidence interval estimator
        
        Args:
            n_iterations: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            random_seed: Random seed for reproducibility
        """
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def compute_ci(self,
                   query_metrics: Dict[str, float],
                   metric_name: str,
                   k: int,
                   aggregation: str = 'macro') -> ConfidenceInterval:
        """
        Compute bootstrap confidence interval
        
        Args:
            query_metrics: Dictionary mapping query_id to metric value
            metric_name: Name of metric (e.g., 'recall', 'precision')
            k: K value
            aggregation: 'macro' (mean) or 'median'
            
        Returns:
            ConfidenceInterval object
        """
        if len(query_metrics) == 0:
            raise ValueError("No query metrics provided")
        
        query_ids = list(query_metrics.keys())
        values = np.array([query_metrics[qid] for qid in query_ids])
        n_queries = len(query_ids)
        
        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(self.n_iterations):
            # Resample queries with replacement
            sample_indices = np.random.choice(n_queries, size=n_queries, replace=True)
            sample_values = values[sample_indices]
            
            # Compute metric for this bootstrap sample
            if aggregation == 'macro':
                bootstrap_means.append(np.mean(sample_values))
            elif aggregation == 'median':
                bootstrap_means.append(np.median(sample_values))
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Compute percentile confidence interval
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        if aggregation == 'macro':
            observed_mean = np.mean(values)
        else:
            observed_mean = np.median(values)
        
        return ConfidenceInterval(
            metric_name=metric_name,
            k=k,
            mean=observed_mean,
            lower=ci_lower,
            upper=ci_upper,
            confidence_level=self.confidence_level,
            n_samples=n_queries
        )
    
    def compute_multiple_ci(self,
                          results_df: pd.DataFrame,
                          metric_name: str,
                          aggregation: str = 'macro') -> List[ConfidenceInterval]:
        """
        Compute CIs for all K values in results DataFrame
        
        Args:
            results_df: DataFrame with columns [query_id, k, metric_name]
            metric_name: Name of metric column
            aggregation: 'macro' or 'median'
            
        Returns:
            List of ConfidenceInterval objects for each K
        """
        cis = []
        
        for k in sorted(results_df['k'].unique()):
            k_data = results_df[results_df['k'] == k]
            query_metrics = dict(zip(k_data['query_id'], k_data[metric_name]))
            
            ci = self.compute_ci(query_metrics, metric_name, k, aggregation)
            cis.append(ci)
        
        return cis


class PermutationTest:
    """
    Permutation test for comparing two ranking systems
    
    Tests null hypothesis: no difference between systems
    """
    
    def __init__(self,
                 n_permutations: int = 1000,
                 alpha: float = 0.05,
                 random_seed: Optional[int] = 42):
        """
        Initialize permutation test
        
        Args:
            n_permutations: Number of permutations
            alpha: Significance level
            random_seed: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def test(self,
            system1_metrics: Dict[str, float],
            system2_metrics: Dict[str, float],
            metric_name: str,
            k: int) -> SignificanceTestResult:
        """
        Perform permutation test
        
        Args:
            system1_metrics: Dictionary mapping query_id to metric (system 1)
            system2_metrics: Dictionary mapping query_id to metric (system 2)
            metric_name: Name of metric
            k: K value
            
        Returns:
            SignificanceTestResult object
        """
        # Ensure same queries
        common_queries = set(system1_metrics.keys()) & set(system2_metrics.keys())
        if len(common_queries) == 0:
            raise ValueError("No common queries between systems")
        
        query_ids = sorted(common_queries)
        values1 = np.array([system1_metrics[qid] for qid in query_ids])
        values2 = np.array([system2_metrics[qid] for qid in query_ids])
        
        # Observed difference
        observed_diff = np.mean(values1) - np.mean(values2)
        
        # Permutation test
        diffs = []
        all_values = np.concatenate([values1, values2])
        n = len(values1)
        
        for _ in range(self.n_permutations):
            # Randomly permute labels
            permuted_indices = np.random.permutation(2 * n)
            perm_group1 = all_values[permuted_indices[:n]]
            perm_group2 = all_values[permuted_indices[n:]]
            
            # Compute difference for this permutation
            perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
            diffs.append(perm_diff)
        
        diffs = np.array(diffs)
        
        # Two-tailed p-value
        p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(values1) + np.var(values2)) / 2)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0.0
        
        return SignificanceTestResult(
            test_name="Permutation Test",
            metric_name=metric_name,
            k=k,
            statistic=observed_diff,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=effect_size
        )


class VarianceAnalyzer:
    """
    Analyze variance and stability of ranking metrics
    """
    
    @staticmethod
    def compute_cv(values: np.ndarray) -> float:
        """
        Compute coefficient of variation
        
        Args:
            values: Array of metric values
            
        Returns:
            Coefficient of variation (std / mean)
        """
        mean = np.mean(values)
        if mean == 0:
            return float('inf')
        return np.std(values) / mean
    
    @staticmethod
    def compute_variance_stats(query_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Compute comprehensive variance statistics
        
        Args:
            query_metrics: Dictionary mapping query_id to metric value
            
        Returns:
            Dictionary with variance statistics
        """
        values = np.array(list(query_metrics.values()))
        
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'var': float(np.var(values)),
            'cv': VarianceAnalyzer.compute_cv(values),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'iqr': float(np.percentile(values, 75) - np.percentile(values, 25))
        }
    
    @staticmethod
    def identify_high_variance_queries(query_metrics: Dict[str, float],
                                       threshold: float = 2.0) -> List[str]:
        """
        Identify queries with unusually low/high metric values
        
        Args:
            query_metrics: Dictionary mapping query_id to metric value
            threshold: Z-score threshold for outliers
            
        Returns:
            List of query IDs with high variance
        """
        values = np.array(list(query_metrics.values()))
        query_ids = list(query_metrics.keys())
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        z_scores = np.abs((values - mean) / std)
        outlier_indices = np.where(z_scores > threshold)[0]
        
        return [query_ids[i] for i in outlier_indices]


def compute_cohens_d(values1: np.ndarray, values2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size
    
    Args:
        values1: Array of values for group 1
        values2: Array of values for group 2
        
    Returns:
        Cohen's d effect size
    """
    mean_diff = np.mean(values1) - np.mean(values2)
    pooled_std = np.sqrt((np.var(values1) + np.var(values2)) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    return mean_diff / pooled_std


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    
    if abs_d < 0.2:
        return "Small effect"
    elif abs_d < 0.5:
        return "Medium effect"
    elif abs_d < 0.8:
        return "Large effect"
    else:
        return "Very large effect"


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("STATISTICAL ANALYSIS EXAMPLE")
    print("=" * 80)
    
    # Simulate query metrics
    np.random.seed(42)
    n_queries = 20
    
    # System 1: Better performance
    system1_metrics = {
        f"Q{i}": np.random.beta(8, 2) for i in range(n_queries)
    }
    
    # System 2: Worse performance
    system2_metrics = {
        f"Q{i}": np.random.beta(6, 4) for i in range(n_queries)
    }
    
    # Bootstrap CI
    print("\n1. Bootstrap Confidence Intervals")
    print("-" * 80)
    bootstrap = BootstrapCI(n_iterations=1000)
    ci = bootstrap.compute_ci(system1_metrics, "recall", k=10)
    print(ci)
    print(f"   Width: {ci.width:.4f}")
    print(f"   Relative width: {ci.relative_width:.2%}")
    
    # Permutation test
    print("\n2. Permutation Test (System 1 vs System 2)")
    print("-" * 80)
    perm_test = PermutationTest(n_permutations=1000)
    result = perm_test.test(system1_metrics, system2_metrics, "recall", k=10)
    print(result)
    print(f"   Effect size (Cohen's d): {result.effect_size:.3f}")
    print(f"   Interpretation: {interpret_cohens_d(result.effect_size)}")
    
    # Variance analysis
    print("\n3. Variance Analysis")
    print("-" * 80)
    var_stats = VarianceAnalyzer.compute_variance_stats(system1_metrics)
    print(f"   Mean: {var_stats['mean']:.4f}")
    print(f"   Std:  {var_stats['std']:.4f}")
    print(f"   CV:   {var_stats['cv']:.4f}")
    print(f"   IQR:  {var_stats['iqr']:.4f}")
