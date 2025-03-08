# load library
import numpy as np
import pandas as pd
import wfdb
import joblib
import glob
import tqdm
from IPython.display import display
from sklearn.model_selection import train_test_split

# load demog data
patients = pd.read_csv('data/mimic-iv-3.1/hosp/patients.csv.gz')
patients['sex'] = patients['gender'].replace({"M":0,"F":1})

admission = pd.read_csv('data/mimic-iv-3.1/hosp/admissions.csv.gz')
patients = patients.merge(admission,how='inner',on='subject_id')
patients['age'] = patients['anchor_age'] + pd.to_datetime(patients['admittime']).dt.year - patients['anchor_year']


# load icd data
dx_cnt = joblib.load("dataset/df/dx_cnt")

# load ecg meta data
ecg_data_df = joblib.load("dataset/df/mimic_iv_ecg_data_df")
ecg_data_df["subject_id"] = ecg_data_df["comments"].str[0].str[14:].astype(int)

# df merge
merge_df = pd.merge(patients,dx_cnt,how='inner',on=['subject_id','hadm_id'])
merge_df = pd.merge(merge_df,ecg_data_df,how='inner',on=['subject_id'])

merge_df["admittime"] = pd.to_datetime(merge_df["admittime"])
merge_df["dischtime"] = pd.to_datetime(merge_df["dischtime"])
merge_df["ecgtime"] = pd.to_datetime(merge_df["base_date"].astype(str) + ' ' + merge_df["base_time"].astype(str))

# The ECG recorded in ER is also taken into account, the window sets 1 day before admission.
merge_df_sel = merge_df.copy()[(merge_df["admittime"] - pd.Timedelta(days=1) < merge_df["ecgtime"]) &
                               (merge_df["ecgtime"] < merge_df["dischtime"])]
merge_df_sel2 = merge_df_sel.sort_values("dischtime").drop_duplicates(subset=['ecgtime'],keep="last")

joblib.dump(merge_df_sel2,"dataset/df/merge_df_sel2")
# 481568 