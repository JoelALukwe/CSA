#!/usr/bin/env python3
"""
Core analysis script:
- Loads preprocessed CSV (samples x cytokines + group)
- Runs PCA, clustering, RandomForest CV, and saves figures + summary table.
Usage:
    python scripts/analysis.py --input data/synthetic_cytokine_preprocessed.csv --outdir results/
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def load_data(path):
    df = pd.read_csv(path, index_col=0)
    return df

def run_analysis(df, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cytokine_cols = [c for c in df.columns if c != 'group']
    X = df[cytokine_cols].values
    y = (df['group'] == 'Active').astype(int).values

    # scaling
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(Xs)

    # Save PCA plot
    plt.figure(figsize=(6,5))
    for g, col in zip(['Latent','Active'], ['blue','red']):
        mask = df['group']==g
        plt.scatter(pcs[mask,0], pcs[mask,1], label=g, alpha=0.8, c=col)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
    plt.title("PCA of Cytokine Data")
    plt.savefig(os.path.join(outdir, "pca_plot.png"), bbox_inches="tight")
    plt.close()

    # Clustering (example)
    cl = AgglomerativeClustering(n_clusters=2)
    labels = cl.fit_predict(Xs)

    # Random Forest CV
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(rf, Xs, y, cv=cv, scoring="roc_auc")
    rf.fit(Xs, y)
    importances = rf.feature_importances_

    # Save feature importance plot
    feat = pd.Series(importances, index=cytokine_cols).sort_values(ascending=True)
    plt.figure(figsize=(6,4))
    feat.plot.barh()
    plt.xlabel("Feature importance")
    plt.title("Random Forest Feature Importances")
    plt.savefig(os.path.join(outdir, "feature_importance.png"), bbox_inches="tight")
    plt.close()

    # Save matrix heatmap-like figure
    plt.figure(figsize=(8,6))
    plt.imshow(StandardScaler().fit_transform(X).T, aspect="auto")
    plt.yticks(range(len(cytokine_cols)), cytokine_cols)
    plt.xlabel("Samples (scaled)")
    plt.title("Scaled Cytokine Matrix")
    plt.savefig(os.path.join(outdir, "scaled_matrix.png"), bbox_inches="tight")
    plt.close()

    # summary CSV
    summary = pd.DataFrame({
        "cytokine": cytokine_cols,
        "importance": importances,
        "pc1_loading": pca.components_[0]
    }).set_index("cytokine")
    summary.to_csv(os.path.join(outdir, "feature_summary.csv"))

    # print quick report
    print("CV AUCs:", aucs)
    print("Saved figures and summary to", outdir)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    df = load_data(args.input)
    run_analysis(df, args.outdir)

if __name__ == "__main__":
    main()
