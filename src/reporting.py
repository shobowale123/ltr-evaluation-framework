"""
Reporting and Visualization Module
Generate comprehensive evaluation reports with visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class EvaluationReporter:
    """
    Generate comprehensive evaluation reports
    
    Combines metrics, statistics, and visualizations into cohesive reports
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize reporter
        
        Args:
            output_dir: Directory to save reports and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def generate_summary_table(self,
                               macro_avg: Dict[int, float],
                               weighted_avg: Dict[int, float],
                               confidence_intervals: List,
                               metric_name: str) -> pd.DataFrame:
        """
        Generate summary metrics table
        
        Args:
            macro_avg: Macro-averaged metrics per K
            weighted_avg: Weighted-averaged metrics per K
            confidence_intervals: List of ConfidenceInterval objects
            metric_name: Name of metric
            
        Returns:
            Summary DataFrame
        """
        ci_dict = {ci.k: ci for ci in confidence_intervals}
        
        rows = []
        for k in sorted(macro_avg.keys()):
            ci = ci_dict.get(k)
            row = {
                'K': k,
                'Metric': f"{metric_name}@{k}",
                'Macro_Avg': macro_avg[k],
                'Weighted_Avg': weighted_avg[k],
                'CI_Lower': ci.lower if ci else np.nan,
                'CI_Upper': ci.upper if ci else np.nan,
                'CI_Width': ci.width if ci else np.nan,
                'CV': np.nan  # To be filled separately
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_stratified_table(self,
                                  stratified_metrics: Dict[str, Dict[int, float]],
                                  stratum_counts: Dict[str, int],
                                  metric_name: str) -> pd.DataFrame:
        """
        Generate stratified analysis table
        
        Args:
            stratified_metrics: Nested dict: stratum -> K -> metric
            stratum_counts: Dictionary mapping stratum to number of queries
            metric_name: Name of metric
            
        Returns:
            Stratified analysis DataFrame
        """
        rows = []
        
        for stratum_name, k_metrics in stratified_metrics.items():
            for k, value in k_metrics.items():
                rows.append({
                    'Stratum': stratum_name,
                    'N_Queries': stratum_counts.get(stratum_name, 0),
                    'K': k,
                    'Metric': f"{metric_name}@{k}",
                    'Value': value
                })
        
        return pd.DataFrame(rows)
    
    def plot_metrics_vs_k(self,
                         results_df: pd.DataFrame,
                         metric_name: str,
                         stratify_by: Optional[str] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot metric values vs K
        
        Args:
            results_df: DataFrame with columns [query_id, k, metric]
            metric_name: Name of metric to plot
            stratify_by: Optional column to stratify by
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if stratify_by and stratify_by in results_df.columns:
            # Stratified plot
            for stratum in results_df[stratify_by].unique():
                stratum_data = results_df[results_df[stratify_by] == stratum]
                k_means = stratum_data.groupby('k')[metric_name].mean()
                ax.plot(k_means.index, k_means.values, marker='o', 
                       label=f"{stratum}", linewidth=2, markersize=8)
        else:
            # Overall plot
            k_means = results_df.groupby('k')[metric_name].mean()
            k_std = results_df.groupby('k')[metric_name].std()
            
            ax.plot(k_means.index, k_means.values, marker='o', 
                   linewidth=2, markersize=8, label='Mean')
            ax.fill_between(k_means.index, 
                           k_means.values - k_std.values,
                           k_means.values + k_std.values,
                           alpha=0.3, label='±1 Std Dev')
        
        ax.set_xlabel('K (Cutoff Position)', fontsize=12)
        ax.set_ylabel(f'{metric_name.capitalize()}@K', fontsize=12)
        ax.set_title(f'{metric_name.capitalize()}@K vs K', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_query_heatmap(self,
                          results_df: pd.DataFrame,
                          metric_name: str,
                          top_n: int = 20,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot heatmap of metric values per query and K
        
        Args:
            results_df: DataFrame with columns [query_id, k, metric]
            metric_name: Name of metric to plot
            top_n: Number of queries to show (top by variance)
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Pivot data
        pivot_df = results_df.pivot(index='query_id', columns='k', values=metric_name)
        
        # Select top N queries by variance
        query_variance = pivot_df.var(axis=1).sort_values(ascending=False)
        top_queries = query_variance.head(top_n).index
        plot_df = pivot_df.loc[top_queries]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.4)))
        
        sns.heatmap(plot_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, cbar_kws={'label': metric_name.capitalize()},
                   ax=ax, linewidths=0.5)
        
        ax.set_xlabel('K (Cutoff Position)', fontsize=12)
        ax.set_ylabel('Query ID', fontsize=12)
        ax.set_title(f'{metric_name.capitalize()}@K Heatmap (Top {top_n} Most Variable Queries)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_difficulty_correlation(self,
                                   query_stats_df: pd.DataFrame,
                                   results_df: pd.DataFrame,
                                   metric_name: str,
                                   k: int,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot metric vs query difficulty
        
        Args:
            query_stats_df: DataFrame with columns [query_id, difficulty_score]
            results_df: DataFrame with columns [query_id, k, metric]
            metric_name: Name of metric to plot
            k: K value to analyze
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Filter for specific K
        k_data = results_df[results_df['k'] == k].copy()
        
        # Merge with difficulty
        merged = k_data.merge(query_stats_df[['query_id', 'difficulty_score']], 
                             on='query_id')
        
        # Compute correlation
        correlation = merged['difficulty_score'].corr(merged[metric_name])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        scatter = ax.scatter(merged['difficulty_score'], merged[metric_name],
                           alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(merged['difficulty_score'], merged[metric_name], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(merged['difficulty_score'].min(), 
                            merged['difficulty_score'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2,
               label=f'Trend (ρ={correlation:.3f})')
        
        ax.set_xlabel('Query Difficulty Score (Neg/Pos Ratio)', fontsize=12)
        ax.set_ylabel(f'{metric_name.capitalize()}@{k}', fontsize=12)
        ax.set_title(f'{metric_name.capitalize()}@{k} vs Query Difficulty', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_comparison(self,
                       system1_metrics: Dict[int, float],
                       system2_metrics: Dict[int, float],
                       system1_name: str = "System 1",
                       system2_name: str = "System 2",
                       metric_name: str = "recall",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison between two systems
        
        Args:
            system1_metrics: Dictionary mapping K to metric for system 1
            system2_metrics: Dictionary mapping K to metric for system 2
            system1_name: Name of system 1
            system2_name: Name of system 2
            metric_name: Name of metric
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        k_values = sorted(system1_metrics.keys())
        sys1_values = [system1_metrics[k] for k in k_values]
        sys2_values = [system2_metrics[k] for k in k_values]
        
        ax.plot(k_values, sys1_values, marker='o', linewidth=2, 
               markersize=8, label=system1_name)
        ax.plot(k_values, sys2_values, marker='s', linewidth=2, 
               markersize=8, label=system2_name)
        
        # Highlight differences
        for k, v1, v2 in zip(k_values, sys1_values, sys2_values):
            if v1 != v2:
                ax.plot([k, k], [v1, v2], 'k--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('K (Cutoff Position)', fontsize=12)
        ax.set_ylabel(f'{metric_name.capitalize()}@K', fontsize=12)
        ax.set_title(f'System Comparison: {metric_name.capitalize()}@K', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_full_report(self,
                           results_df: pd.DataFrame,
                           query_stats_df: pd.DataFrame,
                           macro_avg: Dict[str, Dict[int, float]],
                           weighted_avg: Dict[str, Dict[int, float]],
                           stratified_metrics: Dict[str, Dict[str, Dict[int, float]]],
                           report_name: str = "evaluation_report") -> str:
        """
        Generate comprehensive HTML report
        
        Args:
            results_df: Complete results DataFrame
            query_stats_df: Query statistics DataFrame
            macro_avg: Nested dict: metric -> K -> value (macro)
            weighted_avg: Nested dict: metric -> K -> value (weighted)
            stratified_metrics: Nested dict: metric -> stratum -> K -> value
            report_name: Name for the report
            
        Returns:
            Path to generated HTML report
        """
        report_path = self.output_dir / f"{report_name}.html"
        
        html_parts = []
        html_parts.append(self._html_header(report_name))
        
        # Summary section
        html_parts.append("<h2>1. Executive Summary</h2>")
        html_parts.append(self._generate_summary_section(
            results_df, query_stats_df, macro_avg, weighted_avg))
        
        # Detailed metrics
        html_parts.append("<h2>2. Detailed Metrics</h2>")
        for metric_name in macro_avg.keys():
            html_parts.append(f"<h3>{metric_name.capitalize()}@K</h3>")
            
            # Create summary table
            summary_df = pd.DataFrame({
                'K': sorted(macro_avg[metric_name].keys()),
                'Macro_Avg': [macro_avg[metric_name][k] for k in sorted(macro_avg[metric_name].keys())],
                'Weighted_Avg': [weighted_avg[metric_name][k] for k in sorted(weighted_avg[metric_name].keys())]
            })
            html_parts.append(summary_df.to_html(index=False, float_format='%.4f'))
        
        # Stratified analysis
        html_parts.append("<h2>3. Stratified Analysis</h2>")
        html_parts.append(self._generate_stratified_section(stratified_metrics))
        
        # Query statistics
        html_parts.append("<h2>4. Query Statistics</h2>")
        html_parts.append(query_stats_df.describe().to_html(float_format='%.2f'))
        
        html_parts.append(self._html_footer())
        
        # Write report
        with open(report_path, 'w') as f:
            f.write('\n'.join(html_parts))
        
        return str(report_path)
    
    def _html_header(self, title: str) -> str:
        """Generate HTML header"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric-good {{ color: #27ae60; font-weight: bold; }}
        .metric-poor {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p><em>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
"""
    
    def _html_footer(self) -> str:
        """Generate HTML footer"""
        return """
</body>
</html>
"""
    
    def _generate_summary_section(self, results_df, query_stats_df, macro_avg, weighted_avg) -> str:
        """Generate summary section HTML"""
        summary = []
        summary.append("<div>")
        summary.append(f"<p><strong>Total Queries:</strong> {len(query_stats_df)}</p>")
        summary.append(f"<p><strong>Total Evaluations:</strong> {len(results_df)}</p>")
        
        for metric_name in macro_avg.keys():
            best_k = max(macro_avg[metric_name].items(), key=lambda x: x[1])
            summary.append(f"<p><strong>Best {metric_name.capitalize()}@K:</strong> "
                         f"{best_k[1]:.4f} at K={best_k[0]}</p>")
        
        summary.append("</div>")
        return '\n'.join(summary)
    
    def _generate_stratified_section(self, stratified_metrics) -> str:
        """Generate stratified analysis HTML"""
        html = []
        
        for metric_name, strata in stratified_metrics.items():
            html.append(f"<h3>{metric_name.capitalize()}</h3>")
            
            # Create table
            rows = []
            for stratum_name, k_values in strata.items():
                for k, value in sorted(k_values.items()):
                    rows.append({
                        'Stratum': stratum_name,
                        'K': k,
                        'Value': value
                    })
            
            df = pd.DataFrame(rows)
            if not df.empty:
                html.append(df.to_html(index=False, float_format='%.4f'))
        
        return '\n'.join(html)
    
    def save_results_csv(self, 
                        results_df: pd.DataFrame,
                        filename: str = "detailed_results.csv"):
        """Save detailed results to CSV"""
        output_path = self.output_dir / filename
        results_df.to_csv(output_path, index=False)
        return str(output_path)
    
    def save_summary_json(self,
                         macro_avg: Dict,
                         weighted_avg: Dict,
                         filename: str = "summary_metrics.json"):
        """Save summary metrics to JSON"""
        output_path = self.output_dir / filename
        
        summary = {
            'macro_average': macro_avg,
            'weighted_average': weighted_avg,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(output_path)


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("REPORTING MODULE EXAMPLE")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    queries = [f"Q{i}" for i in range(10)]
    k_values = [5, 10, 20]
    
    data = []
    for q in queries:
        for k in k_values:
            data.append({
                'query_id': q,
                'k': k,
                'recall': np.random.beta(7, 3),
                'precision': np.random.beta(6, 4)
            })
    
    results_df = pd.DataFrame(data)
    
    # Create reporter
    reporter = EvaluationReporter(output_dir="example_results")
    
    # Plot metrics vs K
    print("\nGenerating plots...")
    fig = reporter.plot_metrics_vs_k(results_df, 'recall')
    plt.savefig(reporter.output_dir / "recall_vs_k.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plots to {reporter.output_dir}")
