"""
app.py - Streamlit UI for the Decision Tree project.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split

from hw2 import (
    DecisionTree,
    calc_gini,
    calc_entropy,
    depth_pruning,
    chi_pruning,
    count_nodes,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Decision Tree Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_csv(path_or_buffer, label_col):
    data = pd.read_csv(path_or_buffer).dropna(axis=1)
    if label_col not in data.columns:
        raise ValueError(f"Label column '{label_col}' not found in the dataset.")
    return data


def prepare_arrays(data, label_col, random_state):
    X = data.drop(label_col, axis=1)
    y = data[label_col]
    feature_names = list(X.columns)
    X_arr = np.column_stack([X, y])
    X_train, X_val = train_test_split(X_arr, random_state=int(random_state))
    return X_train, X_val, feature_names


def get_feature_importances(tree, feature_names):
    """Sum feature_importance across all internal nodes, then normalise."""
    importances = {name: 0.0 for name in feature_names}

    def traverse(node):
        if (not node.terminal
                and hasattr(node, "feature_importance")
                and node.feature >= 0
                and node.feature < len(feature_names)):
            importances[feature_names[node.feature]] += node.feature_importance
        for child in node.children:
            traverse(child)

    if tree.root:
        traverse(tree.root)

    total = sum(importances.values())
    if total > 0:
        importances = {k: v / total for k, v in importances.items()}
    return importances


def styled_accuracy(value):
    color = "#4caf50" if value >= 90 else "#ff9800" if value >= 75 else "#f44336"
    return f'<span style="color:{color}; font-weight:700">{value:.2f}%</span>'


def make_line_chart(x, y_series, labels, colors, title, xlabel, ylabel,
                    best_idx=None, best_label=None):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for y, label, color in zip(y_series, labels, colors):
        ax.plot(x, y, marker="o", label=label, color=color, linewidth=2, markersize=5)
    if best_idx is not None and best_label is not None:
        ax.scatter(
            x[best_idx], y_series[1][best_idx],
            s=120, zorder=6, color="#e53935",
            label=f"Best: {best_label}",
        )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Decision Tree Explorer")
    st.caption("Build, evaluate, and prune a Decision Tree from scratch.")
    st.divider()

    # --- Dataset ---
    st.subheader("Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv",
                                     help="Last column should be the class label.")
    label_col = st.text_input("Label column name", value="class")
    random_state = st.number_input("Train / val split seed", value=99, step=1)
    st.divider()

    # --- Model configuration ---
    st.subheader("Model Configuration")

    impurity_choice = st.radio("Impurity function", ["Entropy", "Gini"])
    gain_ratio = st.checkbox("Use gain ratio", value=True,
                             help="Normalises information gain by split information "
                                  "to reduce bias towards high-cardinality features.")

    unlimited = st.checkbox("Unlimited depth", value=True)
    max_depth_slider = st.slider("Max depth", 1, 20, 7, disabled=unlimited)
    max_depth = 1000 if unlimited else max_depth_slider

    chi_val = st.select_slider(
        "Chi-square p-value  (1 = off)",
        options=[1, 0.5, 0.25, 0.1, 0.05, 0.0001],
        value=1,
        help="Lower values prune more aggressively.",
    )
    st.divider()

    col_a, col_b = st.columns(2)
    train_btn      = col_a.button("Train", type="primary",      use_container_width=True)
    experiments_btn = col_b.button("Experiments", use_container_width=True,
                                   help="Run depth and chi-square pruning sweeps.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    if uploaded_file:
        data = load_csv(uploaded_file, label_col)
    else:
        data = load_csv("agaricus-lepiota.csv", label_col)
    data_ok = True
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    data_ok = False

if not data_ok:
    st.stop()

X_train, X_val, feature_names = prepare_arrays(data, label_col, random_state)

# ---------------------------------------------------------------------------
# Main area — four tabs
# ---------------------------------------------------------------------------
tab_data, tab_model, tab_depth, tab_chi = st.tabs(
    ["Dataset", "Model & Results", "Depth Pruning", "Chi-Square Pruning"]
)

# ===========================  Tab 1 – Dataset  ==============================
with tab_data:
    st.header("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total samples",      len(data))
    c2.metric("Features",           len(feature_names))
    c3.metric("Training samples",   len(X_train))
    c4.metric("Validation samples", len(X_val))

    st.divider()
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Sample rows")
        st.dataframe(data.head(20), use_container_width=True, height=300)

    with right:
        st.subheader("Class distribution")
        counts = data[label_col].value_counts()
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        wedge_colors = ["#42a5f5", "#ef5350"]
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=wedge_colors[:len(counts)],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        ax.set_title("Class split", fontsize=11)
        st.pyplot(fig, use_container_width=False)
        plt.close()

        st.subheader("Label counts")
        st.dataframe(counts.rename("count").reset_index(), use_container_width=True)

# =======================  Tab 2 – Model & Results  ==========================
with tab_model:
    st.header("Model & Results")

    if train_btn:
        impurity_func = calc_entropy if impurity_choice == "Entropy" else calc_gini
        with st.spinner("Building decision tree..."):
            tree = DecisionTree(
                data=X_train,
                impurity_func=impurity_func,
                chi=chi_val,
                max_depth=max_depth,
                gain_ratio=gain_ratio,
            )
            tree.build_tree()

        st.session_state["tree"]          = tree
        st.session_state["feature_names"] = feature_names
        st.session_state["X_train"]       = X_train
        st.session_state["X_val"]         = X_val
        st.session_state["config"] = {
            "impurity":    impurity_choice,
            "gain_ratio":  gain_ratio,
            "max_depth":   max_depth,
            "chi":         chi_val,
        }
        st.success("Tree built successfully.")

    if "tree" not in st.session_state:
        st.info("Configure the model in the sidebar and click **Train**.")
        st.stop()

    tree   = st.session_state["tree"]
    fn     = st.session_state["feature_names"]
    Xtr    = st.session_state["X_train"]
    Xvl    = st.session_state["X_val"]
    config = st.session_state["config"]

    # Config recap
    st.caption(
        f"Impurity: **{config['impurity']}**  |  "
        f"Gain ratio: **{config['gain_ratio']}**  |  "
        f"Max depth: **{'unlimited' if config['max_depth'] == 1000 else config['max_depth']}**  |  "
        f"Chi p-value: **{config['chi']}**"
    )

    # Metrics
    train_acc = tree.calc_accuracy(Xtr)
    val_acc   = tree.calc_accuracy(Xvl)
    depth     = tree.depth()
    nodes     = count_nodes(tree.root)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Train accuracy",      f"{train_acc:.2f}%")
    m2.metric("Validation accuracy", f"{val_acc:.2f}%",
              delta=f"{val_acc - train_acc:+.2f}% vs train")
    m3.metric("Tree depth",  depth)
    m4.metric("Total nodes", nodes)

    st.divider()

    # Feature importance
    st.subheader("Feature Importance")
    importances = get_feature_importances(tree, fn)
    sorted_fi = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top_n = 15
    top_features, top_vals = zip(*sorted_fi[:top_n]) if sorted_fi else ([], [])

    if top_vals and sum(top_vals) > 0:
        fig, ax = plt.subplots(figsize=(8, max(3, len(top_features) * 0.38)))
        bars = ax.barh(
            list(reversed(top_features)),
            list(reversed(top_vals)),
            color="#42a5f5",
            edgecolor="white",
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xlabel("Normalised importance")
        ax.set_title(f"Top {top_n} features by importance", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.25, linestyle="--")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.info("Feature importance data not available for this configuration.")

    # Predict a single row
    st.divider()
    st.subheader("Predict a Single Instance")
    row_idx = st.slider("Validation row index", 0, len(Xvl) - 1, 0)
    instance = Xvl[row_idx]
    pred     = tree.predict(instance)
    actual   = instance[-1]
    col_pred, col_actual = st.columns(2)
    col_pred.metric("Prediction", pred)
    col_actual.metric("Actual label", actual,
                      delta="correct" if pred == actual else "wrong",
                      delta_color="normal" if pred == actual else "inverse")

# =======================  Tab 3 – Depth Pruning  ============================
with tab_depth:
    st.header("Depth Pruning Experiment")
    st.caption(
        "Trees are built using Entropy + gain ratio for max_depth values 1 through 10. "
        "Shows how overfitting and underfitting change with tree depth."
    )

    run_depth = experiments_btn or st.button("Run depth pruning", key="btn_depth")

    if run_depth:
        with st.spinner("Training 10 trees..."):
            tr_accs, vl_accs = depth_pruning(X_train, X_val)
        st.session_state["depth_results"] = (tr_accs, vl_accs)

    if "depth_results" not in st.session_state:
        st.info("Click **Run depth pruning** (or **Experiments** in the sidebar).")
    else:
        tr_accs, vl_accs = st.session_state["depth_results"]
        best_d = int(np.argmax(vl_accs)) + 1

        left, right = st.columns([3, 2])

        with left:
            fig = make_line_chart(
                x=list(range(1, 11)),
                y_series=[tr_accs, vl_accs],
                labels=["Training", "Validation"],
                colors=["#42a5f5", "#66bb6a"],
                title="Accuracy vs Max Depth",
                xlabel="Max depth",
                ylabel="Accuracy (%)",
                best_idx=best_d - 1,
                best_label=f"depth={best_d}",
            )
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with right:
            st.subheader("Results table")
            df_depth = pd.DataFrame({
                "Max depth":          range(1, 11),
                "Train acc (%)":      [round(v, 2) for v in tr_accs],
                "Validation acc (%)": [round(v, 2) for v in vl_accs],
            })
            st.dataframe(
                df_depth.style.highlight_max(
                    subset=["Validation acc (%)"], color="#d4edda"
                ),
                use_container_width=True,
                hide_index=True,
                height=380,
            )
            st.success(f"Best validation accuracy at **depth = {best_d}**: {vl_accs[best_d-1]:.2f}%")

# ====================  Tab 4 – Chi-Square Pruning  ==========================
with tab_chi:
    st.header("Chi-Square Pruning Experiment")
    st.caption(
        "Trees are built using Entropy + gain ratio with no depth limit. "
        "Each tree uses a different chi-square significance threshold to control splitting."
    )

    run_chi = experiments_btn or st.button("Run chi-square pruning", key="btn_chi")

    if run_chi:
        with st.spinner("Training 6 trees..."):
            chi_tr, chi_vl, chi_depths = chi_pruning(X_train, X_val)
        st.session_state["chi_results"] = (chi_tr, chi_vl, chi_depths)

    if "chi_results" not in st.session_state:
        st.info("Click **Run chi-square pruning** (or **Experiments** in the sidebar).")
    else:
        chi_tr, chi_vl, chi_depths = st.session_state["chi_results"]
        p_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
        best_i   = int(np.argmax(chi_vl))

        x_labels = [f"p={p}\nd={d}" for p, d in zip(p_values, chi_depths)]

        left, right = st.columns([3, 2])

        with left:
            fig = make_line_chart(
                x=list(range(len(p_values))),
                y_series=[chi_tr, chi_vl],
                labels=["Training", "Validation"],
                colors=["#42a5f5", "#66bb6a"],
                title="Accuracy vs Chi-Square p-value",
                xlabel="",
                ylabel="Accuracy (%)",
                best_idx=best_i,
                best_label=f"p={p_values[best_i]}",
            )
            fig.axes[0].set_xticks(range(len(p_values)))
            fig.axes[0].set_xticklabels(x_labels, fontsize=8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with right:
            st.subheader("Results table")
            df_chi = pd.DataFrame({
                "p-value":            p_values,
                "Tree depth":         chi_depths,
                "Train acc (%)":      [round(v, 2) for v in chi_tr],
                "Validation acc (%)": [round(v, 2) for v in chi_vl],
            })
            st.dataframe(
                df_chi.style.highlight_max(
                    subset=["Validation acc (%)"], color="#d4edda"
                ),
                use_container_width=True,
                hide_index=True,
                height=280,
            )
            st.success(
                f"Best validation accuracy at **p = {p_values[best_i]}** "
                f"(depth = {chi_depths[best_i]}): {chi_vl[best_i]:.2f}%"
            )
