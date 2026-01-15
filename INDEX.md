# 📚 LTR Evaluation Framework - Complete Index

## 🎯 Start Here Based on Your Need

### I want to understand the problem and solution
→ Read [SUMMARY.md](SUMMARY.md) (5 minutes)

### I want to get started quickly
→ Follow [QUICKSTART.md](QUICKSTART.md) (10 minutes)

### I want to understand the methodology
→ Read [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) (30 minutes)

### I want complete documentation
→ Read [README.md](README.md) (15 minutes)

### I want to see working code
→ Run `python examples/complete_evaluation_example.py`

### I want to load my own data
→ Use `examples/load_data_example.py` as template

---

## 📁 Project Structure

```
ltr-evaluation-framework/
│
├── 📖 Documentation
│   ├── README.md                   ← Project overview and features
│   ├── SUMMARY.md                  ← What was built and why
│   ├── QUICKSTART.md              ← 10-minute quick start
│   ├── EVALUATION_METHODOLOGY.md  ← Detailed methodology
│   └── INDEX.md                   ← This file
│
├── 🔧 Core Framework (src/)
│   ├── query_statistics.py        ← Query analysis & stratification
│   ├── k_selector.py              ← Adaptive K selection
│   ├── metrics.py                 ← Recall@K, Precision@K
│   ├── statistical_tests.py       ← Bootstrap CI, significance tests
│   ├── reporting.py               ← Visualization & HTML reports
│   └── pipeline.py                ← End-to-end orchestration
│
├── 💡 Examples & Usage
│   ├── complete_evaluation_example.py  ← Full working example
│   └── load_data_example.py           ← Load data from files
│
└── 📦 Configuration
    └── requirements.txt            ← Dependencies
```

---

## 🚀 Quick Navigation

### By Task

| What I Want to Do | Go To |
|-------------------|-------|
| Understand the problem | [SUMMARY.md](SUMMARY.md) → "Your Challenge" |
| Learn the methodology | [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) |
| Run a quick example | `examples/complete_evaluation_example.py` |
| Load my own data | `examples/load_data_example.py` |
| Understand K selection | [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) → Section 2.1 |
| Learn about metrics | [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) → Section 3 |
| Statistical testing | [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) → Section 4 |
| API documentation | Module docstrings in `src/*.py` |

### By Role

| Your Role | Recommended Path |
|-----------|------------------|
| **Data Scientist** | SUMMARY.md → EVALUATION_METHODOLOGY.md → examples/ |
| **ML Engineer** | QUICKSTART.md → complete_evaluation_example.py → load_data_example.py |
| **Research Scientist** | EVALUATION_METHODOLOGY.md → statistical_tests.py |
| **Manager/Stakeholder** | README.md → SUMMARY.md → View HTML report |
| **PhD Student** | EVALUATION_METHODOLOGY.md (full read) → Implement extensions |

---

## 📊 Output Files Generated

When you run evaluation, you get:

```
evaluation_results/
├── ltr_evaluation_report.html     ← Open this in browser
├── detailed_results.csv            ← Import to Excel/Pandas
├── summary_metrics.json            ← Machine-readable summary
│
├── 📈 Visualizations
│   ├── recall_vs_k.png            ← Recall@K across K values
│   ├── precision_vs_k.png         ← Precision@K across K values
│   ├── recall_heatmap.png         ← Per-query performance heatmap
│   ├── difficulty_correlation_kX.png  ← Performance vs difficulty
│   └── system_comparison.png      ← Model vs baseline comparison
```

---

## 🎓 Learning Path

### Beginner (Just want to use it)
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Run `examples/complete_evaluation_example.py`
3. Modify `examples/load_data_example.py` for your data
4. Generate report and review HTML

### Intermediate (Want to understand)
1. Read [SUMMARY.md](SUMMARY.md)
2. Read [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Sections 1-3
3. Review `src/metrics.py` and `src/k_selector.py`
4. Experiment with different K strategies

### Advanced (Want to extend)
1. Read full [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md)
2. Study `src/statistical_tests.py` and `src/pipeline.py`
3. Review all module docstrings
4. Implement custom K strategies or metrics

---

## 🔑 Key Concepts

### Query Imbalance
Your queries have varying numbers of relevant documents (3 to 120).
→ Solution: **Adaptive K selection** ([EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 2.1)

### Class Imbalance
Highly skewed positive/negative ratios (1:41 to 1:345).
→ Solution: **Query-specific K and proper metrics** ([EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 3)

### Statistical Rigor
Need confidence in results.
→ Solution: **Bootstrap CI and permutation tests** ([EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 4)

### Multiple Perspectives
Different stakeholders care about different things.
→ Solution: **Macro, weighted, and stratified aggregation** ([EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 3.3)

---

## 💻 Code Examples Index

### Basic Usage
```python
# See: examples/complete_evaluation_example.py lines 167-180
pipeline = LTREvaluationPipeline(output_dir="results", k_strategy="adaptive")
pipeline.load_query_statistics(query_data)
pipeline.evaluate_predictions(predictions, ground_truth)
pipeline.generate_full_report()
```

### Load from CSV
```python
# See: examples/load_data_example.py lines 21-50
predictions = load_predictions_from_csv("predictions.csv")
ground_truth = load_ground_truth_from_csv("ground_truth.csv")
```

### Custom K Strategy
```python
# See: src/k_selector.py lines 130-145
config = KConfig(strategy=KStrategy.CUSTOM, custom_k_func=my_func)
selector = KSelector(config)
```

### System Comparison
```python
# See: examples/complete_evaluation_example.py lines 246-254
comparison = pipeline.compare_systems(
    baseline_predictions, ground_truth,
    system1_name="My Model", system2_name="Baseline"
)
```

---

## 📈 Metrics Reference

### Recall@K
**Formula**: `relevant_in_top_K / min(K, n_relevant)`  
**Location**: `src/metrics.py` lines 177-203  
**Doc**: [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 3.1

### Precision@K
**Formula**: `relevant_in_top_K / K`  
**Location**: `src/metrics.py` lines 206-230  
**Doc**: [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 3.2

### Macro-Average
**Formula**: Mean across queries  
**Location**: `src/metrics.py` lines 233-261  
**Doc**: [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 3.3

### Weighted-Average
**Formula**: Weighted by n_relevant  
**Location**: `src/metrics.py` lines 263-292  
**Doc**: [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 3.3

---

## 🔬 Statistical Tests Reference

### Bootstrap Confidence Intervals
**Purpose**: Quantify uncertainty  
**Location**: `src/statistical_tests.py` lines 40-120  
**Doc**: [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 4.1

### Permutation Test
**Purpose**: Compare systems  
**Location**: `src/statistical_tests.py` lines 123-202  
**Doc**: [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 5.2

### Cohen's d Effect Size
**Purpose**: Practical significance  
**Location**: `src/statistical_tests.py` lines 273-299  
**Doc**: [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 5.3

---

## 🎯 Common Tasks

### Task: Evaluate my LTR model
**Steps**:
1. Prepare data (see [QUICKSTART.md](QUICKSTART.md))
2. Run `examples/load_data_example.py`
3. Review HTML report

### Task: Compare two models
**Steps**:
1. Evaluate first model
2. Call `pipeline.compare_systems()` (see [README.md](README.md) → "Advanced Usage")
3. Review significance tests

### Task: Understand why model fails on some queries
**Steps**:
1. Check stratified metrics (HTML report Section 3)
2. Review per-query heatmap (`recall_heatmap.png`)
3. Analyze difficulty correlation (`difficulty_correlation_kX.png`)

### Task: Report to stakeholders
**Steps**:
1. Generate HTML report
2. Share summary table (HTML report Section 1)
3. Highlight stratified analysis (Section 3)

---

## 🆘 Troubleshooting

### Issue: "ValueError: Query ID mismatch"
→ **Fix**: Ensure query IDs match across query_stats, predictions, and ground_truth  
→ **Doc**: [QUICKSTART.md](QUICKSTART.md) → "Common Issues"

### Issue: "Empty ranking"
→ **Fix**: Every query must have at least one predicted document  
→ **Doc**: [QUICKSTART.md](QUICKSTART.md) → "Common Issues"

### Issue: Wide confidence intervals
→ **Meaning**: High variance or insufficient data  
→ **Action**: Check stratified metrics, add more queries, or stabilize model  
→ **Doc**: [SUMMARY.md](SUMMARY.md) → "Interpreting Results"

### Issue: Negative difficulty correlation
→ **Meaning**: Model struggles with imbalanced queries  
→ **Action**: Focus training on imbalanced cases  
→ **Doc**: [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 6.3

---

## 📚 Bibliography & References

Located in [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 10:
- TREC Evaluation Standards
- Bootstrap Methods (Efron & Tibshirani, 1993)
- Information Retrieval (Manning et al.)
- Imbalanced Learning Literature

---

## 🤝 Extension Points

Want to add features? Here are the extension points:

| Feature | Module | Starting Point |
|---------|--------|---------------|
| New K strategy | `k_selector.py` | `KStrategy` enum, `_adaptive_k()` method |
| New metric | `metrics.py` | Add to `QueryEvaluationResult` class |
| New visualization | `reporting.py` | Add plot method to `EvaluationReporter` |
| New statistical test | `statistical_tests.py` | Create new class like `PermutationTest` |

---

## ✅ Pre-Flight Checklist

Before running evaluation:
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Query statistics prepared (CSV with query_id, n_pos, n_neg)
- [ ] Model predictions ready (ranked doc lists per query)
- [ ] Ground truth labels ready (relevant doc sets per query)
- [ ] Query IDs match across all files
- [ ] Output directory has write permissions

---

## 🎓 Terminology

| Term | Definition | Location |
|------|------------|----------|
| **Query imbalance** | Varying n_relevant across queries | [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 1.1 |
| **Class imbalance** | Skewed pos/neg ratio | [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 1.1 |
| **Adaptive K** | Query-specific K selection | [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 2.1 |
| **Macro-average** | Mean across queries | [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 3.3 |
| **Stratum** | Query group by n_relevant | [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 3.3 |
| **Bootstrap CI** | Confidence interval via resampling | [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 4.1 |
| **Cohen's d** | Standardized effect size | [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) Section 5.3 |

---

## 📞 Support & Contribution

- **Questions**: Review documentation first, then open issue
- **Bugs**: Check [QUICKSTART.md](QUICKSTART.md) → "Common Issues", then report
- **Features**: Propose via issue with use case
- **Documentation**: Improvements always welcome

---

## 🎉 Quick Wins

1. **5 minutes**: Read [SUMMARY.md](SUMMARY.md) to understand the solution
2. **10 minutes**: Run `python examples/complete_evaluation_example.py`
3. **30 minutes**: Adapt `examples/load_data_example.py` for your data
4. **1 hour**: Generate your first evaluation report
5. **2 hours**: Read [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) and become expert

---

## 📊 Your Data Quick Reference

From your provided data:

```
Queries: 24
Positives: 3 to 120 (median=28)
Negatives: 510 to 4966 (median=1376)
Imbalance: 1:41 to 1:345

Stratification:
- Low (3-10):  8 queries
- Medium (11-50): 10 queries  
- High (51+):  6 queries

Recommended K strategy: ADAPTIVE
```

---

*Built with ❤️ for rigorous IR evaluation*

**Version**: 1.0.0  
**Last Updated**: January 15, 2026  
**Status**: Production Ready ✅
