#!/usr/bin/env python3
"""
Generate a realistic synthetic cytokine dataset (samples x cytokines + group column).
Usage:
    python scripts/generate_synthetic_data.py --output data/synthetic_cytokine.csv --n 120
"""
import numpy as np
import pandas as pd
import argparse

def generate(n=120, seed=42):
    np.random.seed(seed)
    cytokines = ["IFNg","IL2","TNFa","IL10","IL6","IL1b","IL8","IL4","IL12p70","IL17A"]
    half = n // 2
    labels = ["Latent"]*half + ["Active"]*(n-half)
    data = []
    for i in range(n):
        base = np.random.lognormal(mean=1.5, sigma=0.6, size=len(cytokines))
        if labels[i] == "Active":
            base[0] *= 1.6  # IFNg
            base[2] *= 1.4  # TNFa
            base[4] *= 1.3  # IL6
            base += np.random.normal(0, 0.1, size=base.shape)
        data.append(base)
    df = pd.DataFrame(data, columns=cytokines)
    df.index = [f"sample_{i+1}" for i in range(n)]
    df["group"] = labels
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=120)
    args = p.parse_args()
    df = generate(n=args.n)
    df.to_csv(args.output)
    print(f"Wrote synthetic data to {args.output}")

if __name__ == "__main__":
    main()
