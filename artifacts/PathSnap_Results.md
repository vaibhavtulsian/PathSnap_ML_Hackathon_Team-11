# PathSnap Results

## Dataset
- Source: `features.parquet`
- Total candidate records: 2,881,651
- Feature columns: 16
- Training sample: 40,000
- Testing sample: 10,000
- Test labels: 9,304 negative and 696 positive

## Bayesian Logistic Regression Results
The Bayesian Logistic Regression model was trained using PyMC ADVI.

| Metric | Score |
|---|---:|
| Accuracy | 94.94% |
| Precision | 90.95% |
| Recall | 30.32% |
| F1-score | 45.47% |
| ROC-AUC | 96.01% |

### Classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 95% | 100% | 97% | 9,304 |
| 1 | 91% | 30% | 45% | 696 |
| Macro average | 93% | 65% | 71% | 10,000 |
| Weighted average | 95% | 95% | 94% | 10,000 |

## Feature Coefficients

| Feature | Mean coefficient |
|---|---:|
| arc_length | 0.184099 |
| speed | 0.038353 |
| heading_valid | 0.028774 |
| frac_along | 0.015341 |
| d_next | -0.046532 |
| d_prev | -0.046640 |
| heading_diff_undirected | -0.094360 |
| n_cand | -0.141606 |
| dist_rel_min | -0.352328 |
| rank | -0.471372 |
| distance | -0.488869 |
| heading_diff | -0.680214 |

## Additional Output
- Top-1 candidate accuracy reported by the notebook: 7.12%.
- Number of GPS points in that evaluation: 9,746.

## Interpretation
The model achieved high overall accuracy and ROC-AUC, but positive-class recall was relatively low (30.32%). The dataset is imbalanced, with 9,304 negative samples and 696 positive samples in the test set. Therefore, accuracy alone does not fully describe model performance; precision, recall, F1-score, and ROC-AUC should also be considered.

## Notes
These results are taken from the uploaded notebook outputs. The notebook's reported Top-1 Candidate Accuracy (7.12%) is a separate evaluation output and should not be conflated with the binary classification accuracy (94.94%).
