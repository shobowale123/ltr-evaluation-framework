"""
Query Statistics Module
Analyzes query-level characteristics for imbalanced LTR evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class QueryStratum(Enum):
    """Query stratification based on relevance count"""
    LOW = "Low (3-10 relevant)"
    MEDIUM = "Medium (11-50 relevant)"
    HIGH = "High (51+ relevant)"


@dataclass
class QueryStats:
    """Statistics for a single query"""
    query_id: str
    n_pos: int
    n_neg: int
    n_total: int
    imbalance_ratio: float
    pos_ratio: float
    stratum: QueryStratum
    difficulty_score: float
    
    def __repr__(self):
        return (f"QueryStats(id={self.query_id}, pos={self.n_pos}, "
                f"neg={self.n_neg}, difficulty={self.difficulty_score:.2f})")


class QueryStatisticsAnalyzer:
    """
    Comprehensive query-level statistics analyzer
    
    Handles:
    - Query imbalance characterization
    - Class imbalance quantification
    - Stratification by query characteristics
    - Difficulty score computation
    """
    
    def __init__(self, query_data: pd.DataFrame):
        """
        Initialize analyzer with query-level data
        
        Args:
            query_data: DataFrame with columns [query_id, n_pos, n_neg]
        """
        required_cols = ['query_id', 'n_pos', 'n_neg']
        if not all(col in query_data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        self.data = query_data.copy()
        self.data['n_total'] = self.data['n_pos'] + self.data['n_neg']
        self.data['imbalance_ratio'] = self.data['n_neg'] / self.data['n_pos']
        self.data['pos_ratio'] = self.data['n_pos'] / self.data['n_total']
        
        # Compute statistics
        self._compute_statistics()
        
    def _compute_statistics(self):
        """Compute comprehensive statistics"""
        self.stats = {
            'n_queries': len(self.data),
            'pos_stats': self._compute_column_stats('n_pos'),
            'neg_stats': self._compute_column_stats('n_neg'),
            'total_stats': self._compute_column_stats('n_total'),
            'imbalance_stats': self._compute_column_stats('imbalance_ratio'),
            'pos_ratio_stats': self._compute_column_stats('pos_ratio')
        }
        
    def _compute_column_stats(self, col: str) -> Dict[str, float]:
        """Compute statistics for a column"""
        values = self.data[col]
        return {
            'min': float(values.min()),
            'max': float(values.max()),
            'mean': float(values.mean()),
            'median': float(values.median()),
            'std': float(values.std()),
            'q25': float(values.quantile(0.25)),
            'q75': float(values.quantile(0.75)),
            'cv': float(values.std() / values.mean()) if values.mean() > 0 else 0.0
        }
    
    def get_query_stats(self, query_id: str) -> QueryStats:
        """
        Get comprehensive statistics for a specific query
        
        Args:
            query_id: Query identifier
            
        Returns:
            QueryStats object with all query characteristics
        """
        row = self.data[self.data['query_id'] == query_id]
        if len(row) == 0:
            raise ValueError(f"Query {query_id} not found")
        
        row = row.iloc[0]
        
        return QueryStats(
            query_id=query_id,
            n_pos=int(row['n_pos']),
            n_neg=int(row['n_neg']),
            n_total=int(row['n_total']),
            imbalance_ratio=float(row['imbalance_ratio']),
            pos_ratio=float(row['pos_ratio']),
            stratum=self._get_stratum(int(row['n_pos'])),
            difficulty_score=float(row['imbalance_ratio'])
        )
    
    def _get_stratum(self, n_pos: int) -> QueryStratum:
        """Assign query to stratum based on n_pos"""
        if n_pos <= 10:
            return QueryStratum.LOW
        elif n_pos <= 50:
            return QueryStratum.MEDIUM
        else:
            return QueryStratum.HIGH
    
    def get_all_query_stats(self) -> List[QueryStats]:
        """Get statistics for all queries"""
        return [self.get_query_stats(qid) for qid in self.data['query_id']]
    
    def stratify_queries(self) -> Dict[QueryStratum, List[str]]:
        """
        Stratify queries by relevance count
        
        Returns:
            Dictionary mapping stratum to list of query IDs
        """
        stratification = {stratum: [] for stratum in QueryStratum}
        
        for qid in self.data['query_id']:
            stats = self.get_query_stats(qid)
            stratification[stats.stratum].append(qid)
        
        return stratification
    
    def get_stratum_statistics(self) -> pd.DataFrame:
        """
        Get aggregate statistics per stratum
        
        Returns:
            DataFrame with stratum-level statistics
        """
        stratification = self.stratify_queries()
        
        results = []
        for stratum, query_ids in stratification.items():
            stratum_data = self.data[self.data['query_id'].isin(query_ids)]
            
            results.append({
                'stratum': stratum.value,
                'n_queries': len(query_ids),
                'avg_n_pos': stratum_data['n_pos'].mean(),
                'avg_n_neg': stratum_data['n_neg'].mean(),
                'avg_imbalance': stratum_data['imbalance_ratio'].mean(),
                'min_n_pos': stratum_data['n_pos'].min(),
                'max_n_pos': stratum_data['n_pos'].max()
            })
        
        return pd.DataFrame(results)
    
    def get_summary_report(self) -> str:
        """
        Generate comprehensive summary report
        
        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 80)
        report.append("QUERY STATISTICS SUMMARY")
        report.append("=" * 80)
        report.append(f"\nTotal Queries: {self.stats['n_queries']}")
        
        report.append("\n" + "-" * 80)
        report.append("POSITIVE (RELEVANT) DOCUMENTS")
        report.append("-" * 80)
        self._add_stats_to_report(report, self.stats['pos_stats'])
        
        report.append("\n" + "-" * 80)
        report.append("NEGATIVE (NON-RELEVANT) DOCUMENTS")
        report.append("-" * 80)
        self._add_stats_to_report(report, self.stats['neg_stats'])
        
        report.append("\n" + "-" * 80)
        report.append("IMBALANCE RATIO (Neg/Pos)")
        report.append("-" * 80)
        self._add_stats_to_report(report, self.stats['imbalance_stats'])
        
        report.append("\n" + "-" * 80)
        report.append("STRATIFICATION")
        report.append("-" * 80)
        stratum_df = self.get_stratum_statistics()
        report.append("\n" + stratum_df.to_string(index=False))
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def _add_stats_to_report(self, report: List[str], stats: Dict[str, float]):
        """Add statistics dictionary to report"""
        report.append(f"  Min:        {stats['min']:.2f}")
        report.append(f"  Max:        {stats['max']:.2f}")
        report.append(f"  Mean:       {stats['mean']:.2f}")
        report.append(f"  Median:     {stats['median']:.2f}")
        report.append(f"  Std Dev:    {stats['std']:.2f}")
        report.append(f"  Q25:        {stats['q25']:.2f}")
        report.append(f"  Q75:        {stats['q75']:.2f}")
        report.append(f"  CV:         {stats['cv']:.3f}")
    
    def get_difficulty_distribution(self) -> pd.DataFrame:
        """
        Get distribution of query difficulty scores
        
        Returns:
            DataFrame with difficulty statistics per query
        """
        return self.data[['query_id', 'n_pos', 'n_neg', 'imbalance_ratio']].rename(
            columns={'imbalance_ratio': 'difficulty_score'}
        ).sort_values('difficulty_score', ascending=False)
    
    def identify_outlier_queries(self, 
                                 method: str = 'iqr',
                                 threshold: float = 1.5) -> List[str]:
        """
        Identify outlier queries based on imbalance ratio
        
        Args:
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or z-score threshold
            
        Returns:
            List of outlier query IDs
        """
        if method == 'iqr':
            q1 = self.data['imbalance_ratio'].quantile(0.25)
            q3 = self.data['imbalance_ratio'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = self.data[
                (self.data['imbalance_ratio'] < lower_bound) | 
                (self.data['imbalance_ratio'] > upper_bound)
            ]
        elif method == 'zscore':
            mean = self.data['imbalance_ratio'].mean()
            std = self.data['imbalance_ratio'].std()
            z_scores = np.abs((self.data['imbalance_ratio'] - mean) / std)
            outliers = self.data[z_scores > threshold]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return outliers['query_id'].tolist()


def load_query_data_from_text(text_data: str) -> pd.DataFrame:
    """
    Parse query data from tab-separated text
    
    Args:
        text_data: String with format "query_id\\tn_pos\\tn_neg" per line
        
    Returns:
        DataFrame with columns [query_id, n_pos, n_neg]
    """
    lines = [line.strip() for line in text_data.strip().split('\n') if line.strip()]
    
    data = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) == 3:
            data.append({
                'query_id': parts[0].strip(),
                'n_pos': int(parts[1].strip()),
                'n_neg': int(parts[2].strip())
            })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage with provided data
    sample_data = """
04-CWP-005\t78\t3243
05-3P-001C\t66\t1859
05-3P-002\t28\t1331
05-CWP-011\t15\t1219
06-1P-004B\t45\t1648
06-1P-008F\t120\t4966
06-1P-009\t3\t1036
06-1P-010\t3\t549
06-1P-014D\t120\t2969
06-1P-015\t3\t510
06-3P-001\t78\t2115
06-3P-006\t10\t1706
06-3P-008\t28\t1289
06-5P-007\t28\t1910
07-1P-004\t10\t641
07-1P-006\t55\t1776
07-1P-007\t15\t1066
07-1P-010\t15\t1060
07-2P-005\t10\t1086
07-2P-006\t36\t2318
07-2P-007\t36\t1421
07-2P-009\t6\t590
07-3P-001\t28\t2750
07-3P-002\t3\t837
"""
    
    df = load_query_data_from_text(sample_data)
    analyzer = QueryStatisticsAnalyzer(df)
    
    print(analyzer.get_summary_report())
