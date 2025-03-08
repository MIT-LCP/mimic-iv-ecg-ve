# File and directory handling
import os, random, glob, joblib
# Data manipulation
import pandas as pd
import numpy as np
# Scikit-learn for machine learning and metrics
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, recall_score, precision_score, fbeta_score, confusion_matrix



# df_with_ve = joblib.load("dataset/df/test_with_ve_df")
merge_df = joblib.load("dataset/df/annotation_df.jb")
merge_df = merge_df.drop('filename',axis=1)
merge_df = merge_df.rename(columns={'filename':'ecg_path', "af":"af_ai"})

merge_df_sel4 = joblib.load("dataset/df/merge_df_sel4")
ve = pd.read_csv("embeddings/embedding_128dim.csv")

merge_df_sel4['filename'] = "dataset/tfdata/tftest/ecg_tf_" + merge_df_sel4['ecg_pID'] +"_" + merge_df_sel4['ecg_sID']
df_with_ve = pd.merge(merge_df_sel4,ve,how="inner", on=["filename"])


df_with_ve_with_af = pd.merge(merge_df, df_with_ve,how="inner", on=["ecg_sID","ecg_pID"])

df_with_ve_with_af_sel = df_with_ve_with_af.copy()[df_with_ve_with_af['annot_Dx'].isin(["AF","nonAF"])]

df_with_ve_with_af_sel["af_dr"] = df_with_ve_with_af_sel['annot_Dx'].replace({"AF":1,"nonAF":0})


latent_dim = 128
VE_vars = ['VE_' + str(item).zfill(2) for item in range(latent_dim)]

# Preparation of five-part cross-validation per patient
gkf = GroupKFold(n_splits=5)

# Create pipeline (preprocessing + model)
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('lasso_logreg', LogisticRegression(penalty='l2', solver='saga', max_iter=10000, C=10)) 
])
# Create custom metrics
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

scoring = {
    'AUC': 'roc_auc',
    'Sensitivity': make_scorer(recall_score),
    'Specificity': make_scorer(specificity_score),
    'Precision': make_scorer(precision_score),
    'F2': make_scorer(fbeta_score, beta=2)
}

def mean_sd(series,digit):
    return str(round(np.mean(series),digit)).zfill(digit)+' ± '+ str(round(
        np.std(series),digit)).zfill(digit)

def five_cv_eval(df, outcome):
    cv_results = cross_validate(pipeline, df[VE_vars],
                            df[outcome], cv=gkf.split(df[VE_vars],
                            df[outcome], groups=df.subject_id), scoring=scoring,
                            n_jobs=-1 )
    print("outcome: ", outcome)
    print("AUC: ", mean_sd(cv_results['test_AUC'],2))
    print("Sensitivity: ", mean_sd(cv_results['test_Sensitivity'],2))          
    print("Specificity: ", mean_sd(cv_results['test_Specificity'],2))   
    print("Precision: ", mean_sd(cv_results['test_Precision'],2))   
    print("F2 score: ", mean_sd(cv_results['test_F2'],2))   


five_cv_eval(df_with_ve_with_af_sel,"af_ai")
five_cv_eval(df_with_ve_with_af_sel,'af_dr')

five_cv_eval(df_with_ve_with_af,"hf")
five_cv_eval(df_with_ve_with_af,"sex")
five_cv_eval(df_with_ve_with_af,"af_ai")



# outcome:  af_ai
# AUC:  0.84 ± 0.01
# Sensitivity:  0.52 ± 0.02
# Specificity:  0.94 ± 0.0
# Precision:  0.82 ± 0.02
# F2 score:  0.56 ± 0.02
# outcome:  af_dr
# AUC:  0.83 ± 0.01
# Sensitivity:  0.46 ± 0.02
# Specificity:  0.94 ± 0.01
# Precision:  0.79 ± 0.02
# F2 score:  0.5 ± 0.02


# outcome:  hf
# AUC:  0.72 ± 0.0
# Sensitivity:  0.22 ± 0.01
# Specificity:  0.95 ± 0.0
# Precision:  0.63 ± 0.03
# F2 score:  0.25 ± 0.01
# outcome:  sex
# AUC:  0.63 ± 0.01
# Sensitivity:  0.57 ± 0.01
# Specificity:  0.63 ± 0.01
# Precision:  0.58 ± 0.01
# F2 score:  0.57 ± 0.01
# outcome:  af_ai
# AUC:  0.76 ± 0.01
# Sensitivity:  0.13 ± 0.01
# Specificity:  0.99 ± 0.0
# Precision:  0.58 ± 0.04
# F2 score:  0.15 ± 0.02