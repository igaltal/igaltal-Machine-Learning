# Decision Tree from Scratch

A Decision Tree classifier built from scratch in Python, without using sklearn's tree implementation. Supports Gini and Entropy impurity measures, gain ratio splitting, depth pruning, and chi-square pre-pruning. Comes with an interactive web UI and a command-line runner. Applied to the UCI Mushroom dataset to classify mushrooms as edible or poisonous.

---

## Quickstart

### 1. Install dependencies

```
pip install numpy pandas matplotlib scikit-learn streamlit
```

### 2. Launch the web UI

```
streamlit run app.py
```

This opens the app in your browser automatically. That is the recommended way to use this project.

### 3. Or run from the command line

```
python run.py
```

---

## Running the web UI

```
streamlit run app.py
```

The sidebar lets you configure everything. The main area has four tabs:

| Tab | What it shows |
|-----|---------------|
| Dataset | Row count, class distribution pie chart, data preview |
| Model & Results | Accuracy metrics, feature importance chart, single-row prediction |
| Depth Pruning | Accuracy vs max depth (1-10), results table |
| Chi-Square Pruning | Accuracy vs p-value, tree depth per configuration, results table |

**Sidebar controls:**

| Control | Description |
|---------|-------------|
| Upload CSV | Use your own dataset. Last column must be the class label. Leave empty to use the default mushroom dataset. |
| Label column name | Name of the target column (default: `class`) |
| Train / val split seed | Random seed for reproducibility |
| Impurity function | Entropy or Gini |
| Use gain ratio | Normalises information gain to reduce bias towards high-cardinality features |
| Unlimited depth | When checked, the tree grows until all leaves are pure |
| Max depth | Active only when unlimited depth is unchecked. Range 1-20. |
| Chi-square p-value | Significance threshold for pre-pruning. `1` disables pruning entirely. Lower values produce smaller trees. |
| Train | Builds a single tree with the current configuration |
| Experiments | Runs both the depth pruning and chi-square pruning sweeps |

---

## Running from the command line

```
python run.py
```

Options:

```
python run.py                        # full evaluation: compare all configs + both pruning sweeps
python run.py --depth-only           # depth pruning experiment only
python run.py --chi-only             # chi-square pruning experiment only
python run.py --data path/to/file    # use a different CSV file
python run.py --no-plots             # skip matplotlib plots (useful in headless environments)
```

Plots are saved as `depth_pruning.png` and `chi_pruning.png` in the working directory.

---

## Using the API directly

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from hw2 import DecisionTree, calc_gini, calc_entropy, depth_pruning, chi_pruning, count_nodes

# Load data (last column must be the label)
data = pd.read_csv('agaricus-lepiota.csv').dropna(axis=1)
X, y = data.drop('class', axis=1), data['class']
X = np.column_stack([X, y])
X_train, X_val = train_test_split(X, random_state=99)

# Build a tree
tree = DecisionTree(data=X_train, impurity_func=calc_entropy, gain_ratio=True)
tree.build_tree()

# Evaluate
print(tree.calc_accuracy(X_train))   # training accuracy (%)
print(tree.calc_accuracy(X_val))     # validation accuracy (%)
print(tree.depth())                  # max depth of the tree
print(count_nodes(tree.root))        # total number of nodes

# Predict a single instance
prediction = tree.predict(X_val[0])

# Pruning experiments
train_accs, val_accs = depth_pruning(X_train, X_val)           # sweep max_depth 1-10
chi_train, chi_val, depths = chi_pruning(X_train, X_val)       # sweep 6 p-values
```

---

## Project structure

```
.
├── app.py                  # Streamlit web UI
├── run.py                  # Command-line runner
├── hw2.py                  # Core implementation
├── hw2.ipynb               # Jupyter notebook with step-by-step walkthrough
└── agaricus-lepiota.csv    # UCI Mushroom dataset
```

### hw2.py — what is implemented

| Component | Description |
|-----------|-------------|
| `calc_gini(data)` | Gini impurity of a dataset |
| `calc_entropy(data)` | Entropy of a dataset |
| `DecisionNode` | A single node in the tree. Handles splitting, chi-square pruning, and feature importance. |
| `DecisionTree` | Wraps the root node. Builds the tree, predicts, and calculates accuracy. |
| `depth_pruning(X_train, X_val)` | Trains 10 trees with max_depth 1-10, returns accuracy lists |
| `chi_pruning(X_train, X_val)` | Trains 6 trees with different chi p-values, returns accuracy lists and depths |
| `count_nodes(node)` | Counts total nodes in a tree |

---

## DecisionTree parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | np.ndarray | required | Training data. Last column must be the class label. |
| `impurity_func` | function | required | `calc_gini` or `calc_entropy` |
| `chi` | float | `1` | Chi-square p-value threshold. `1` = no pruning. Options: `1`, `0.5`, `0.25`, `0.1`, `0.05`, `0.0001` |
| `max_depth` | int | `1000` | Maximum tree depth. `1000` is effectively unlimited. |
| `gain_ratio` | bool | `False` | Use gain ratio instead of raw information gain when selecting the best split feature |

---

## Dataset

The UCI Mushroom dataset contains 8124 samples from 23 species of gilled mushrooms. Each sample has 21 categorical features (cap shape, odor, gill color, etc.) and a binary label: edible or poisonous. The dataset is split 75/25 into training (6093 samples) and validation (2031 samples).

---

## Results summary

| Configuration | Train accuracy | Validation accuracy |
|---------------|---------------|---------------------|
| Gini, gain ratio off | ~99.4% | ~77.5% |
| Entropy, gain ratio off | ~99.5% | ~77.3% |
| Entropy, gain ratio on | ~99.8% | ~78.5% |

Entropy with gain ratio gives the best validation accuracy and is used as the default in all pruning experiments.
