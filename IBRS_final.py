
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

from SAD_IBRS.data_load import load_data
from SAD_IBRS.PLS_Reg import pls_imp



def select_features_elasticnet(X: pd.DataFrame,
                               y: pd.Series,
                               seed: int = 0,
                               l1_ratios: list[float] = [.3,.5,.7,.9],
                               cv: int = 5,
                               min_features: int | None = 1,
                               max_features: int | None = 4
                               ) -> list[str]:
    """
    ElasticNetCV -> SelectFromModel
    - max_features: SelectFromModel에서 바로 제한
    - min_features: 선택 수가 부족하면 |coef| 상위로 채움
    """
    # ElasticNetCV 학습
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('enet',   ElasticNetCV(l1_ratio=l1_ratios,
                                alphas=None,
                                cv=cv,
                                random_state=RANDOM_STATE))
    ])
    pipe.fit(X, y)
    model = pipe.named_steps['enet']

    # SelectFromModel (max_features 제한)
    selector = SelectFromModel(model, prefit=True,
                               threshold='mean',          # |coef| ≥ 평균
                               max_features=max_features) # 최대 개수
    mask = selector.get_support()
    selected = list(X.columns[mask])

    # min_features 보장: 부족하면 상위 |coef|로 추가
    if min_features and len(selected) < min_features:
        coef_abs = pd.Series(np.abs(model.coef_), index=X.columns)
        extra = coef_abs.sort_values(ascending=False).index.difference(selected)
        need  = min_features - len(selected)
        selected += list(extra[:need])

    print(f'ElasticNet selected {len(selected)} features.')
    return selected



def corr_and_print(x, y, tag):
    r, p = stats.spearmanr(x, y)
    print(f'{tag}: r={r:.3f}, p={p:.4g}')
    return r, p



    
def run_pipeline():

    mods, PRS, LSAS, group = load_data(num_roi=156)

    print('\n[Stage-1] ElasticNet-Logistic -> IBRS')

    X = pd.concat(list(mods.values()), axis=1)
    y = group

    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    ibrs  = np.zeros(len(y))
    beta_s_list = []  

    for fold, (tr, te) in tqdm(enumerate(outer.split(X, y), 1)):
        # 모달리티별 ElasticNet 선택
        X_tr_sel, X_te_sel = [], []
        for mod_name, X_mod in mods.items():
            print(f'{mod_name}: start ElasticNet')
            sel = select_features_elasticnet(X_mod.iloc[tr], y.iloc[tr], seed=RANDOM_STATE)
            X_tr_sel.append(X_mod.iloc[tr][sel])
            X_te_sel.append(X_mod.iloc[te][sel])

        X_tr_all = pd.concat(X_tr_sel, axis=1)
        X_te_all = pd.concat(X_te_sel, axis=1)

        # 로짓 ElasticNet
        print('logit ElasticNet')
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                penalty='elasticnet', solver='saga',
                l1_ratios=[.1,.3,.5,.7,.9],
                Cs=np.logspace(-3, 2, 40), cv=5,
                scoring='roc_auc', max_iter=5000,
                random_state=RANDOM_STATE
            )
        )
        pipe.fit(X_tr_all, y.iloc[tr])
        ibrs[te] = pipe.decision_function(X_te_all)

        print(f'Fold{fold:2d} AUC={roc_auc_score(y.iloc[te], ibrs[te]):.3f}')

        # coef를 Series로 저장
        coef = pipe.named_steps['logisticregressioncv'].coef_.ravel()
        beta_s_list.append(pd.Series(coef, index=X_tr_all.columns))

    # fold 간 계수 정렬 & 평균
    beta_df = pd.concat(beta_s_list, axis=1).fillna(0) 
    beta_df.columns = [f'fold{i+1}' for i in range(beta_df.shape[1])]
    # 평균 |β| 중요도
    beta_mean = beta_df.abs().mean(axis=1).sort_values(ascending=False)

    beta_df.to_csv(ARTIFACT_DIR / 'elasticnet_coefficients_by_fold.csv')
    beta_mean.to_csv(ARTIFACT_DIR / 'elasticnet_coefficients_mean_abs.csv')   
    pd.Series(ibrs, index=X.index, name='IBRS').to_csv(ARTIFACT_DIR / 'ibrs_oof_scores.csv')

    print(f'\nOOF AUC={roc_auc_score(y, ibrs):.3f}')
    print('\nTop-5 β (|mean|)\n', beta_mean.head())

    # IBRS로 PRS/LSAS 회귀
    Y = pd.concat([PRS, LSAS], axis=1)
    reg = LinearRegression().fit(Y, ibrs.reshape(-1,1))
    print('R2 (IBRS -> [PRS,LSAS]) =', reg.score(Y,ibrs.reshape(-1,1)))


    print('IBRS 비선형상관(spearman)')
    corr_and_print(ibrs, PRS, 'IBRS~PRS')
    corr_and_print(ibrs, LSAS, 'IBRS~LSAS')

    # Pearson
    r_prs, p_prs = pearsonr(ibrs, PRS)
    r_lsas, p_lsas = pearsonr(ibrs, LSAS)
    print(f"IBRS~PRS: r={r_prs:.3f}, p={p_prs:.3g}")
    print(f"IBRS~LSAS: r={r_lsas:.3f}, p={p_lsas:.3g}")

    # R2 / RMSE
    for name, y_cont in [("PRS", PRS), ("LSAS", LSAS)]:
        lr = LinearRegression().fit(ibrs.reshape(-1,1), y_cont)
        y_pred = lr.predict(ibrs.reshape(-1,1))
        print(f"{name}  R2 = {r2_score(y_cont, y_pred):.3f}, RMSE = {root_mean_squared_error(y_cont, y_pred):.3f}")


    # 최종 파이프라인 재학습
    pipe_final = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(penalty='elasticnet', solver='saga',
                            l1_ratios=[.3,.5,.7,.9],
                            Cs=np.logspace(-2,1,20),
                            cv=5, scoring='roc_auc',
                            max_iter=5000, n_jobs=-1,
                            random_state=RANDOM_STATE)
    )
    # 모달리티별 1차 선택을 통과한 전체 ROI 집합 중 10-fold 50% 이상 나타난 ROI만 선택
    stability = (beta_df != 0).astype(int).sum(axis=1) / beta_df.shape[1]
    selected_features = stability[stability >= 0.5].index  
    pipe_final.fit(X[selected_features], y)


    print('\n[Stage-2] PLSRegression')

    # beta_df: ROI × fold coef 테이블 (NaN=0)
    print(f'Stable ROI = {len(stable_roi)}개')
    X_pls = X[selected_features] 
    Y = pd.DataFrame({'PRS': PRS, 'LSAS': LSAS})

    # selected_features에 대해 pls 분석
    save_dir = ARTIFACT_DIR / 'ibrs_features'
    save_dir.mkdir(parents=True, exist_ok=True)
    pls_imp(X_pls,Y,pipe_final,save_dir)

    # 전체 features에 대해 pls 분석
    save_dir = ARTIFACT_DIR / 'all_features'
    save_dir.mkdir(parents=True, exist_ok=True)
    pls_imp(X,Y,pipe_final,save_dir)



if __name__ == '__main__':

    RANDOM_STATE = 626
    # OUTER_FOLDS = 10

    BASE_DIR = Path(f"/your/path/ibrs")
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    ARTIFACT_DIR = BASE_DIR / "artifacts"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    run_pipeline(num_roi=156)