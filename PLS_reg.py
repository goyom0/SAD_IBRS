
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






def subtype_latent_analysis(pls_pipe, X, PRS, LSAS,
                            n_latent: int = 2,  n_clusters: int = 4,
                            seed: int = 0):

    # 잠재 점수 추출
    x_lat = pls_pipe.named_steps['plsregression'].x_scores_[:, :n_latent]

    # 2D 임베딩 (UMAP)
    reducer = UMAP(n_components=2, random_state=RANDOM_STATE)
    emb = reducer.fit_transform(x_lat)

    # GMM 클러스터
    gmm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)
    cluster = gmm.fit_predict(x_lat)
    print(f'Silhouette={silhouette_score(x_lat, cluster):.3f}')

    # 잠재 점수 ↔ 임상 상관
    for tgt, name in [(PRS, 'PRS'), (LSAS, 'LSAS')]:
        r, p = stats.pearsonr(x_lat[:, 0], tgt)
        print(f'LV1~{name}: r={r:.3f}, p={p:.4g}')

    # 클러스터별 PRS·LSAS 요약
    for k in range(n_clusters):
        mask = cluster == k
        print(f'Cluster{k}: n={mask.sum()} PRS μ={PRS[mask].mean():.2f} LSAS μ={LSAS[mask].mean():.1f}')

    # 클러스터 시각화
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(6, 5))
    # scatter = ax.scatter(emb[:, 0], emb[:, 1], c=cluster, s=30, alpha=.8, cmap='tab10')
    # ax.set(title='UMAP of PLS latent space', xlabel='UMAP1', ylabel='UMAP2')
    # legend1 = ax.legend(*scatter.legend_elements(), title='Subtype')
    # ax.add_artist(legend1)
    # plt.tight_layout(); plt.close()

    # 클러스터별 PRS 분포 히스토그램
    # fig, ax = plt.subplots(figsize=(6, 4))
    # bins = np.linspace(PRS.min(), PRS.max(), 20)
    # for k in range(n_clusters):
    #     ax.hist(PRS[cluster == k], bins=bins, alpha=.4, label=f'C{k}')
    # ax.set(title='PRS distribution by subtype', xlabel='PRS', ylabel='count')
    # ax.legend(); plt.tight_layout(); plt.close()

    return cluster




def pls_imp(X: pd.DataFrame,
            Y: pd.DataFrame,
            pipe: Pipeline,            # 리그레션 파이프라인 인자 추가
            save_dir: Path):
    # PLS fitting
    pls = make_pipeline(StandardScaler(), PLSRegression(n_components=2))
    pls.fit(X, Y)
    x_lat = pls.named_steps['plsregression'].x_scores_[:, 0]
    y_lat = pls.named_steps['plsregression'].y_scores_[:, 0]
    latent_r = float(np.corrcoef(x_lat, y_lat)[0, 1])
    print(f'Latent corr={latent_r:.3f}')

    # PLS loadings & SHAP
    load = pd.Series(pls.named_steps['plsregression'].x_loadings_[:, 0],
                        index=X.columns)
    imp_pls = load.abs().sort_values(ascending=False)

    # SHAP 계산을 위한 transform
    scaler = pipe.named_steps['standardscaler']
    feat_in = list(scaler.feature_names_in_)  
    X_for_shap = X.reindex(columns=feat_in)     

    if X_for_shap.isna().any().any():
        missing = set(feat_in) - set(X.columns)
        raise ValueError(f'Missing features for SHAP transform: {missing}')
    X_scaled = scaler.transform(X_for_shap)

    expl = shap.LinearExplainer(pipe.named_steps['logisticregressioncv'],
                                X_scaled,  check_additivity=False)
    sv = expl.shap_values(X_scaled)
    imp_shap = pd.Series(np.abs(sv).mean(0),
                        index=feat_in).sort_values(ascending=False)

    top = pd.DataFrame({'|PLS loading|': imp_pls.head(15),
                        'mean |SHAP|':  imp_shap.head(15)})

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'latent_corr.json').write_text(json.dumps({'LV1_corr': latent_r}, indent=2))
    imp_pls.to_csv( save_dir / 'pls_loadings_abs.csv')
    imp_shap.to_csv(save_dir / 'shap_importance_mean_abs.csv')
    top.to_csv(save_dir / 'top15_pls_vs_shap.csv')
    pd.DataFrame({'x_lat': x_lat, 'y_lat': y_lat},index=X.index).to_csv(save_dir / 'latent_scores.csv')

    # 파이프라인 저장
    # joblib.dump(pls, save_dir / 'pls_pipeline.joblib')
    # joblib.dump(pipe, save_dir / 'final_logit_pipeline.joblib')
    print(f'All outputs saved”')

    # 클러스터 확인
    cluster = subtype_latent_analysis(pls, X, PRS.values, LSAS.values,
                            n_latent=2, n_clusters=4, seed=RANDOM_STATE)
    try:
        pd.DataFrame(cluster, index=X.index, columns=['cluster']).to_csv(save_dir / 'cluster.csv')
        # 클러스터별 PLS LV1 boxplot
        fig, ax = plt.subplots(figsize=(5, 4))
        groups = [x_lat[cluster == k] for k in np.unique(cluster)]
        labels = [f'C{k}' for k in np.unique(cluster)]
        ax.boxplot(groups, labels=labels)
        ax.set(title='PLS LV1 by Subtype', ylabel='LV1 score')
        plt.tight_layout()
        plt.savefig(save_dir / 'cluster4_box.png', dpi=300, bbox_inches="tight")
        plt.close()
    except: pass