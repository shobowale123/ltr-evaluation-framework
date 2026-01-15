# LTR System Evaluation Methodology
## Data-Driven Approach for Imbalanced IR Systems

**Author**: Senior Data Scientist  
**Date**: January 15, 2026  
**Problem**: Evaluate Learning-to-Rank (LTR) system with query and class imbalance

---

## 1. Problem Statement

### 1.1 System Characteristics
- **Training Data**: `[query_id, doc_1, doc_2, ..., doc_n, feature_1, ..., feature_m]`
- **Evaluation Data**: `[query_id, n_pos, n_neg]` - Aggregated relevance counts
- **Target Metrics**: Recall@K, Precision@K
- **Challenges**:
  - Query imbalance: 3 to 120 relevant documents per query
  - Class imbalance: Varying pos/neg ratios (1:170 to 1:41)
  - Variable K per query based on relevance counts

### 1.2 Data Characteristics

| Metric | Min | Max | Median | Mean | Std Dev |
|--------|-----|-----|--------|------|---------|
| Positives | 3 | 120 | 28 | 38.5 | 34.1 |
| Negatives | 510 | 4966 | 1421 | 1673.3 | 1065.8 |
| Total Docs | 513 | 5086 | 1449 | 1711.8 | 1089.5 |
| Imbalance Ratio | 1:41 | 1:170 | 1:51 | 1:58 | - |

---

## 2. Evaluation Framework Design

### 2.1 Query-Specific K Selection Strategy

Given the imbalance, we need **multiple K values per query**:

#### Strategy 1: Percentile-Based K
For each query q with n_pos relevant documents:
- **K = [10%, 25%, 50%, 75%, 100%] of n_pos**
- Minimum: K_min = max(1, n_pos * 0.1)
- Maximum: K_max = n_pos

Example for query `06-1P-014D` (n_pos=120):
- K = [12, 30, 60, 90, 120]

#### Strategy 2: Fixed Cutoffs with Query Cap
Use standard IR cutoffs, capped by query capacity:
- **K_standard = [5, 10, 20, 50, 100]**
- **K_actual = min(K_standard, n_pos)**

#### Strategy 3: Adaptive K Based on Distribution
For queries with:
- **Low positives (n_pos < 10)**: K = [1, 3, n_pos]
- **Medium positives (10 ≤ n_pos < 50)**: K = [5, 10, 20, n_pos]
- **High positives (n_pos ≥ 50)**: K = [10, 20, 50, n_pos]

**Recommendation**: Use **Strategy 3** as it adapts to query characteristics while maintaining comparability.

---

## 3. Metrics Definition

### 3.1 Recall@K (Query-Specific)

For query q with n_pos relevant documents at cutoff K:

$$\text{Recall@K}_q = \frac{\text{# relevant docs in top K}}{\min(K, n_{pos,q})}$$

**Why this formulation?**
- When K ≥ n_pos: Perfect recall = 1.0 (retrieved all possible relevant docs)
- When K < n_pos: Measures fraction of relevant docs retrieved

### 3.2 Precision@K (Query-Specific)

$$\text{Precision@K}_q = \frac{\text{# relevant docs in top K}}{K}$$

### 3.3 Aggregation Strategies

#### Macro-Average (Equal Weight Per Query)
$$\text{Metric}_{\text{macro}} = \frac{1}{|Q|} \sum_{q \in Q} \text{Metric}_q$$

**Use when**: All queries equally important regardless of size

#### Weighted Average (By Relevance Count)
$$\text{Metric}_{\text{weighted}} = \frac{\sum_{q \in Q} n_{pos,q} \cdot \text{Metric}_q}{\sum_{q \in Q} n_{pos,q}}$$

**Use when**: Larger queries more important (common in production)

#### Stratified Analysis
Group queries by n_pos and compute metrics per stratum:
- **Low**: n_pos ∈ [3, 10]
- **Medium**: n_pos ∈ (10, 50]
- **High**: n_pos > 50

---

## 4. Statistical Methodology

### 4.1 Confidence Intervals

Use **bootstrap resampling** (query-level) to compute 95% CI:
1. Resample queries with replacement (1000 iterations)
2. Compute metric for each bootstrap sample
3. Use percentile method: [2.5%, 97.5%] quantiles

### 4.2 Variance Analysis

Compute **coefficient of variation** to assess metric stability:
$$CV = \frac{\sigma(\text{Metric})}{\mu(\text{Metric})}$$

High CV indicates high variance across queries → need stratified analysis

### 4.3 Query Difficulty Analysis

For each query, compute:
- **Query Difficulty Score**: $\text{Difficulty} = \frac{n_{neg}}{n_{pos}}$
- Correlate with performance: $\rho(\text{Difficulty}, \text{Recall@K})$

Identifies if system struggles with highly imbalanced queries

---

## 5. Comparison Framework

### 5.1 Baseline Comparison

Compare LTR model against:
1. **Random Baseline**: Random ranking
2. **BM25 Baseline**: Traditional IR baseline
3. **Oracle**: Perfect ranking (upper bound)

### 5.2 Statistical Significance Testing

Use **permutation test** (query-level):
- Null hypothesis: No difference between models
- Test statistic: Difference in mean metric
- p-value: Proportion of permutations with ≥ observed difference
- Significance level: α = 0.05

### 5.3 Effect Size

Compute **Cohen's d** to measure practical significance:
$$d = \frac{\mu_1 - \mu_2}{\sigma_{pooled}}$$

Interpretation:
- |d| < 0.2: Small effect
- 0.2 ≤ |d| < 0.5: Medium effect
- |d| ≥ 0.5: Large effect

---

## 6. Evaluation Protocol

### 6.1 Data Preparation

```python
# For each query:
# 1. Load model predictions (ranked list)
# 2. Load ground truth (relevance labels)
# 3. Compute query statistics
# 4. Determine K values for query
```

### 6.2 Metric Computation Pipeline

```
For each query q:
  1. Get top-K predictions from model
  2. Compare with ground truth labels
  3. Compute Recall@K and Precision@K for all K values
  4. Store query-level results

Aggregate:
  5. Compute macro-average across queries
  6. Compute weighted average by relevance
  7. Compute stratified metrics
  8. Generate confidence intervals
  9. Compute variance statistics
```

### 6.3 Reporting Structure

**Primary Metrics Table**:
| Metric | K | Macro | Weighted | 95% CI | CV |
|--------|---|-------|----------|--------|-----|
| Recall | Adaptive | X.XXX | X.XXX | [X.XX, X.XX] | X.XX |
| Precision | Adaptive | X.XXX | X.XXX | [X.XX, X.XX] | X.XX |

**Stratified Analysis Table**:
| Stratum | # Queries | Recall@K | Precision@K |
|---------|-----------|----------|-------------|
| Low (3-10) | N | X.XXX | X.XXX |
| Medium (11-50) | N | X.XXX | X.XXX |
| High (51+) | N | X.XXX | X.XXX |

**Query Difficulty Correlation**:
- Spearman ρ(Difficulty, Recall@K)
- Identify struggling query ranges

---

## 7. Key Considerations for Imbalanced Data

### 7.1 Don't Use
❌ **Accuracy**: Misleading with class imbalance  
❌ **F1-Score at fixed K**: Not appropriate for varying relevance counts  
❌ **Micro-averaging**: Over-weights large queries  

### 7.2 Do Use
✅ **Recall@K**: Captures retrieval completeness  
✅ **Precision@K**: Captures ranking quality  
✅ **Stratified analysis**: Reveals performance patterns  
✅ **Query-specific K**: Respects data distribution  
✅ **Confidence intervals**: Quantifies uncertainty  

### 7.3 Warning Signs
- High CV (>0.5): Performance varies widely across queries
- Negative correlation with difficulty: Model struggles with imbalance
- Stratified performance gaps: Model biased toward certain query types

---

## 8. Implementation Checklist

- [ ] Load model predictions and ground truth
- [ ] Compute query statistics (n_pos, n_neg, difficulty)
- [ ] Determine adaptive K values per query
- [ ] Compute Recall@K and Precision@K per query
- [ ] Aggregate using macro and weighted averaging
- [ ] Generate confidence intervals via bootstrap
- [ ] Perform stratified analysis
- [ ] Compute correlation with query difficulty
- [ ] Compare against baselines with significance tests
- [ ] Generate comprehensive report with visualizations

---

## 9. Expected Outputs

1. **Metrics CSV**: Query-level results for all metrics and K values
2. **Summary Report**: Aggregated metrics with confidence intervals
3. **Stratified Analysis**: Performance by query characteristics
4. **Visualizations**:
   - Recall@K vs K per stratum
   - Precision@K vs K per stratum
   - Performance vs query difficulty scatter plot
   - Per-query metric heatmap
5. **Statistical Tests**: Significance and effect sizes vs baselines

---

## 10. References & Best Practices

- **TREC Evaluation Standards**: Query-level metric aggregation
- **Bootstrap Methods**: Efron & Tibshirani (1993)
- **IR Evaluation**: Manning et al., "Introduction to Information Retrieval"
- **Imbalanced Learning**: Handling class imbalance in ranking
