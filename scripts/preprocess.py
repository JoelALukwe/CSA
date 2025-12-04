#!/usr/bin/env python3
"""
Preprocessing utilities for cytokine datasets.
Assumptions:
- Input CSV has samples in rows, cytokine columns, and a `group` column (e.g., 'Latent'/'Active').
- Index optional.
Usage:
    python scripts/preprocess.py --input data/synthetic_cytokine.csv --output data/synthetic_cytokine_preprocessed.csv
"""
import pandas as pd
import numpy as np
import argparse
from sklearn.impute import SimpleImputer

def load_data(path):
    df = pd.read_csv(path, index_col=0)
    return df

def basic_qc(df):
    report = {}
    report['shape'] = df.shape
    report['missing_per_column'] = df.isna().sum().to_dict()
    return report

def impute_and_transform(df, cytokine_cols, strategy='median', log_transform=True):
    df = df.copy()
    imp = SimpleImputer(strategy=strategy)
    X = df[cytokine_cols].values
    X_imputed = imp.fit_transform(X)
    X_df = pd.DataFrame(X_imputed, columns=cytokine_cols, index=df.index)
    if log_transform:
        X_df = np.log1p(X_df)
    # merge group back
    out = df.copy()
    out[cytokine_cols] = X_df
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--strategy", default="median")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    df = load_data(args.input)
    # Identify cytokine columns (all except 'group')
    if 'group' not in df.columns:
        raise ValueError("Input CSV must contain a 'group' column.")
    cytokine_cols = [c for c in df.columns if c != 'group']
    print("QC:", basic_qc(df))
    df_p = impute_and_transform(df, cytokine_cols, strategy=args.strategy, log_transform=args.log)
    # Standardize column order
    df_p.to_csv(args.output)
    print(f"Wrote preprocessed data to {args.output}")

if __name__ == "__main__":
    main()
