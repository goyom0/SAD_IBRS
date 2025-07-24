
### 단순회귀로 결과 빨리 내기..

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.metrics import root_mean_squared_error as sklearn_mse
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from sklearn.model_selection import ShuffleSplit ,StratifiedGroupKFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_decomposition import PLSRegression
from typing import Dict, Any
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from umap import UMAP
import scipy.stats as stats
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import SelectFromModel
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import r2_score, root_mean_squared_error
import joblib, json


# 잔차화 및 스케일링
def residualize(X: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    model = LinearRegression().fit(demo, X)
    preds = model.predict(demo)
    resid = X.values - preds
    resid = pd.DataFrame(StandardScaler().fit_transform(resid), index=X.index, columns=X.columns)
    return resid




def load_data(num_roi: int = 156):
    
    # Load your data
    fc_array, alff_array, reho_array, roi_vol_array, demo_df, _ = prep_data(num_roi=num_roi)
    
    lut = pd.read_csv('/your/path/atlas.tsv', sep='\t', index_col=0)

    demo_df = demo_df.set_index("ID")
    demo_df["sex"] = demo_df["sex"].apply(lambda x: 1 if x == 2 else 0)
    demo_df["group"] = demo_df["group"].apply(lambda x: 1 if x == 'EXP' else 0)
    demo_df["age_std"] = (demo_df["age"] - demo_df["age"].mean()) / demo_df["age"].std()

    # roi_vol_array.index = demo_df.index
    # roi_vol_array.columns = roi_vol_array.columns.str.replace("-", "_", regex=False)

    # thickness_array = roi_vol_array[roi_vol_array.columns[roi_vol_array.columns.str.endswith('Thick')]]
    # surf_array = roi_vol_array[roi_vol_array.columns[roi_vol_array.columns.str.endswith('Surf')]]
    # vol_array = roi_vol_array[roi_vol_array.columns[roi_vol_array.columns.str.endswith('Vol')]]

    PRS = (demo_df['PRS']  - demo_df['PRS'].mean())  / demo_df['PRS'].std()
    LSAS = (demo_df['LSAS'] - demo_df['LSAS'].mean()) / demo_df['LSAS'].std()
    group = demo_df["group"]
    demo_vars = demo_df[["age_std", "sex"]]

    # alff_array.columns = [f'{col}_alff' for col in lut['label']]
    # reho_array.columns = [f'{col}_reho' for col in lut['label']]
    # vol_array.columns = [f'r{col}' for col in vol_array.columns]
    # thickness_array.columns = [f'{col}' for col in thickness_array.columns]
    # surf_array.columns = [f'{col}' for col in surf_array.columns]

    mods: Dict[str, pd.DataFrame] = {
        # "fc": residualize(fc_array, demo_vars),
        "alff": residualize(alff_array, demo_vars),
        "reho": residualize(reho_array, demo_vars),
        "vol": residualize(vol_array, demo_vars),
        "thick": residualize(thickness_array, demo_vars),
        # "surf": residualize(surf_array, demo_vars),
    }

    return mods, PRS, LSAS, group
