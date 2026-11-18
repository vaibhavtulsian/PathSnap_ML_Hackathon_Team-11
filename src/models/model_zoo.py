"""
PathSnap: 5+ Model Experimentation and Bayesian Optimization.
Owned by Member 2 (Models and Tuning).

Required outputs:
- Comparison table across 5 distinct algorithm families
- Hyperparameter tuning via Bayesian Optimization (Optuna)
- Saved best model weights in artifacts/
- Sections 3 & 4 of Technical Report
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def get_candidate_models() -> Dict[str, Any]:
    """
    Returns 5 distinct model families as required by Hackathon guidelines:
    1. Linear/Probabilistic: Logistic Regression
    2. Instance-based: KNN
    3. Tree/Ensemble: Random Forest
    4. Ensemble/Extremely Randomized: Extra Trees
    5. Neural Network: Multi-layer Perceptron (MLP)
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    }
