"""
run.py - Command-line runner for the Decision Tree project.

Usage:
    python run.py                        # run full evaluation with default settings
    python run.py --depth-only           # only run depth pruning experiment
    python run.py --chi-only             # only run chi-square pruning experiment
    python run.py --data path/to/file    # use a different CSV file
    python run.py --no-plots             # skip matplotlib plots
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hw2 import (
    DecisionTree,
    calc_gini,
    calc_entropy,
    depth_pruning,
    chi_pruning,
    count_nodes,
)


def load_data(csv_path, random_state=99):
    data = pd.read_csv(csv_path).dropna(axis=1)
    X, y = data.drop("class", axis=1), data["class"]
    X = np.column_stack([X, y])
    X_train, X_val = train_test_split(X, random_state=random_state)
    print(f"Loaded {csv_path}")
    print(f"  Training samples  : {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    return X_train, X_val


def compare_impurity_functions(X_train, X_val):
    print("\n--- Impurity Function Comparison ---")
    configs = [
        ("Gini,           gain_ratio=False", calc_gini,    False),
        ("Entropy,        gain_ratio=False", calc_entropy, False),
        ("Entropy,        gain_ratio=True ", calc_entropy, True),
    ]
    best_val_acc = -1
    best_label = None
    for label, func, gr in configs:
        tree = DecisionTree(data=X_train, impurity_func=func, gain_ratio=gr)
        tree.build_tree()
        train_acc = tree.calc_accuracy(X_train)
        val_acc   = tree.calc_accuracy(X_val)
        nodes     = count_nodes(tree.root)
        depth     = tree.depth()
        print(f"  {label}  train={train_acc:.2f}%  val={val_acc:.2f}%  "
              f"nodes={nodes}  depth={depth}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_label = label
    print(f"\n  Best configuration: {best_label.strip()}  (val={best_val_acc:.2f}%)")


def run_depth_pruning(X_train, X_val, show_plots):
    print("\n--- Depth Pruning (max_depth 1-10) ---")
    train_accs, val_accs = depth_pruning(X_train, X_val)
    best_depth = int(np.argmax(val_accs)) + 1
    print(f"  {'depth':>5}  {'train %':>8}  {'val %':>8}")
    for d, (tr, va) in enumerate(zip(train_accs, val_accs), start=1):
        marker = " <-- best" if d == best_depth else ""
        print(f"  {d:>5}  {tr:>8.2f}  {va:>8.2f}{marker}")

    if show_plots:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, 11), train_accs, label="Training")
        plt.plot(range(1, 11), val_accs, label="Validation")
        plt.scatter(best_depth, val_accs[best_depth - 1], c="red", zorder=5,
                    label=f"Best depth={best_depth}")
        plt.xlabel("Max Depth")
        plt.ylabel("Accuracy (%)")
        plt.title("Depth Pruning")
        plt.legend()
        plt.tight_layout()
        plt.savefig("depth_pruning.png", dpi=100)
        print("  Plot saved to depth_pruning.png")
        plt.show()


def run_chi_pruning(X_train, X_val, show_plots):
    print("\n--- Chi-Square Pruning ---")
    p_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    train_accs, val_accs, depths = chi_pruning(X_train, X_val)
    best_idx = int(np.argmax(val_accs))
    print(f"  {'p-value':>8}  {'depth':>6}  {'train %':>8}  {'val %':>8}")
    for p, d, tr, va in zip(p_values, depths, train_accs, val_accs):
        marker = " <-- best" if va == val_accs[best_idx] else ""
        print(f"  {p:>8}  {d:>6}  {tr:>8.2f}  {va:>8.2f}{marker}")

    if show_plots:
        import matplotlib.pyplot as plt
        labels = [str((p, d)) for p, d in zip(p_values, depths)][::-1]
        plt.figure()
        plt.plot(labels, train_accs[::-1], label="Training")
        plt.plot(labels, val_accs[::-1], label="Validation")
        best_label = labels[len(p_values) - 1 - best_idx]
        plt.scatter(best_label, val_accs[best_idx], c="red", zorder=5,
                    label=f"Best p={p_values[best_idx]}")
        plt.xlabel("(p-value, tree depth)")
        plt.ylabel("Accuracy (%)")
        plt.title("Chi-Square Pruning")
        plt.xticks(rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig("chi_pruning.png", dpi=100)
        print("  Plot saved to chi_pruning.png")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Decision Tree runner")
    parser.add_argument("--data",       default="agaricus-lepiota.csv",
                        help="Path to the CSV dataset (default: agaricus-lepiota.csv)")
    parser.add_argument("--depth-only", action="store_true",
                        help="Run only the depth pruning experiment")
    parser.add_argument("--chi-only",   action="store_true",
                        help="Run only the chi-square pruning experiment")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Disable matplotlib plots")
    args = parser.parse_args()

    show_plots = not args.no_plots
    X_train, X_val = load_data(args.data)

    if not args.depth_only and not args.chi_only:
        compare_impurity_functions(X_train, X_val)
        run_depth_pruning(X_train, X_val, show_plots)
        run_chi_pruning(X_train, X_val, show_plots)
    elif args.depth_only:
        run_depth_pruning(X_train, X_val, show_plots)
    elif args.chi_only:
        run_chi_pruning(X_train, X_val, show_plots)


if __name__ == "__main__":
    main()
