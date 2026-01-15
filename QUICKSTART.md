# Quick Start Guide
## Getting Your LTR System Evaluated in 10 Minutes

### Step 1: Install Dependencies (1 minute)

```bash
cd /workspaces/ltr-evaluation-framework
pip install -r requirements.txt
```

### Step 2: Prepare Your Data (2 minutes)

You need three pieces of data:

#### A. Query Statistics (what you already have)
```
query_id	n_pos	n_neg
04-CWP-005	78	3243
05-3P-001C	66	1859
...
```

#### B. Model Predictions (ranked list per query)
Format: For each query, a list of document IDs ranked by your model

```python
from src.metrics import QueryPredictions

predictions = {
    "04-CWP-005": QueryPredictions(
        query_id="04-CWP-005",
        ranked_doc_ids=["doc123", "doc456", "doc789", ...]  # Top to bottom
    ),
    # ... more queries
}
```

#### C. Ground Truth (which docs are relevant)
```python
from src.metrics import QueryGroundTruth

ground_truth = {
    "04-CWP-005": QueryGroundTruth(
        query_id="04-CWP-005",
        relevant_doc_ids={"doc123", "doc789", ...}  # Set of relevant doc IDs
    ),
    # ... more queries
}
```

### Step 3: Run Evaluation (5 seconds)

```python
from src.pipeline import LTREvaluationPipeline
from src.query_statistics import load_query_data_from_text

# Load your query stats
query_data = load_query_data_from_text("""
04-CWP-005	78	3243
05-3P-001C	66	1859
...
""")

# Initialize
pipeline = LTREvaluationPipeline(
    output_dir="my_results",
    k_strategy="adaptive"  # Handles imbalance automatically
)

# Load stats
pipeline.load_query_statistics(query_data)

# Evaluate (predictions and ground_truth from Step 2)
pipeline.evaluate_predictions(predictions, ground_truth)

# Get results
pipeline.generate_full_report()
```

### Step 4: Review Results (2 minutes)

Check `my_results/` folder:
- `ltr_evaluation_report.html` - Open this in browser
- `detailed_results.csv` - Import into Excel/Pandas
- `*.png` - Visualizations

### What You Get

#### Console Output
```
Recall@K:
  K=  5: 0.8234
  K= 10: 0.9012
  K= 20: 0.9345

Stratified Analysis:
  Low (3-10 rel):    Recall@10 = 0.9123
  Medium (11-50 rel): Recall@10 = 0.8734
  High (51+ rel):     Recall@10 = 0.7891
```

#### HTML Report
Beautiful report with:
- Summary metrics table
- Stratified analysis
- Visualizations embedded
- Statistical significance

### Example: Run the Demo

```bash
cd examples
python complete_evaluation_example.py
```

This runs evaluation on simulated data matching your dataset characteristics.

---

## Common Issues

### "No module named 'src'"
```bash
# Make sure you're in the project root
cd /workspaces/ltr-evaluation-framework

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/workspaces/ltr-evaluation-framework"
```

### "Query ID mismatch"
Make sure query IDs match across:
- Query statistics
- Predictions
- Ground truth

### "Empty ranking"
Every query must have at least one predicted document.

---

## Integration with Your Workflow

### Loading from Files

#### CSV Format
```python
import pandas as pd

# Query stats
query_data = pd.read_csv("query_stats.csv")  # columns: query_id, n_pos, n_neg

# Predictions (CSV with query_id, doc_id, rank)
pred_df = pd.read_csv("predictions.csv")
predictions = {}
for qid, group in pred_df.groupby('query_id'):
    ranked_docs = group.sort_values('rank')['doc_id'].tolist()
    predictions[qid] = QueryPredictions(qid, ranked_docs)

# Ground truth (CSV with query_id, doc_id)
gt_df = pd.read_csv("ground_truth.csv")
ground_truth = {}
for qid, group in gt_df.groupby('query_id'):
    relevant = set(group['doc_id'])
    ground_truth[qid] = QueryGroundTruth(qid, relevant)
```

#### JSON Format
```python
import json

with open("predictions.json") as f:
    pred_data = json.load(f)
    predictions = {
        qid: QueryPredictions(qid, data['ranked_docs'])
        for qid, data in pred_data.items()
    }
```

### Automated Pipeline

```python
def evaluate_model(model_name, predictions_file, ground_truth_file):
    """Automated evaluation wrapper"""
    
    # Load query stats
    query_data = load_query_data_from_text(QUERY_STATS)
    
    # Initialize pipeline
    pipeline = LTREvaluationPipeline(
        output_dir=f"results/{model_name}",
        k_strategy="adaptive"
    )
    
    # Load and evaluate
    pipeline.load_query_statistics(query_data)
    predictions = load_predictions(predictions_file)
    ground_truth = load_ground_truth(ground_truth_file)
    pipeline.evaluate_predictions(predictions, ground_truth)
    
    # Generate report
    pipeline.generate_full_report(f"{model_name}_evaluation")
    
    # Return key metrics
    aggregates = pipeline.compute_aggregate_metrics()
    return aggregates['recall']['macro']

# Use it
recall_scores = evaluate_model("my_ltr_model", "preds.csv", "truth.csv")
print(f"Recall@10: {recall_scores[10]:.4f}")
```

---

## Next Steps

1. **Replace simulation with real data** - Use your actual model predictions
2. **Compare multiple models** - Use `pipeline.compare_systems()`
3. **Tune based on strata** - Focus on low-performing query types
4. **Automate in CI/CD** - Add to your model evaluation pipeline

---

## Need Help?

- Read [EVALUATION_METHODOLOGY.md](EVALUATION_METHODOLOGY.md) for deep dive
- Check [examples/](examples/) for more patterns
- Review module docstrings for API details
