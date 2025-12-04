# Cytokine Signature Analysis - Notebook (script form)
# 1) Load libraries and data
import pandas as pd
from pathlib import Path
p = Path(__file__).parents[1]
data_path = p / "data" / "synthetic_cytokine.csv"
df = pd.read_csv(data_path, index_col=0)
df.head()

# 2) Preprocess (impute + log1p)
from sklearn.impute import SimpleImputer
import numpy as np
imp = SimpleImputer(strategy="median")
cytokines = [c for c in df.columns if c!='group']
X_imp = imp.fit_transform(df[cytokines])
X_imp = np.log1p(X_imp)
# continue with PCA and plots (see scripts/analysis.py for reproducible code)
