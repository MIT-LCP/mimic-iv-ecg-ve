# load library
import numpy as np
import pandas as pd
import wfdb
import joblib
import glob
import tqdm
from IPython.display import display
from sklearn.model_selection import train_test_split

icd_df = pd.read_csv('data/mimic-iv-3.1/hosp/d_icd_diagnoses.csv.gz')


# https://www.cms.gov/icd10m/version36-fullcode-cms/fullcode_cms/P0467.html
# https://en.wikipedia.org/wiki/List_of_ICD-9_codes_390%E2%80%93459:_diseases_of_the_circulatory_system#:~:text=428%20Heart%20failure&text=428.4%20Heart%20failure%2C%20combined%2C%20unspec.
# https://www.bcbsnm.com/provider/education-reference/education/news/2021-archive/02-15-2021-atrial-fibrillation
# https://en.wikipedia.org/wiki/List_of_ICD-9_codes_390%E2%80%93459:_diseases_of_the_circulatory_system#:~:text=428%20Heart%20failure&text=428.4%20Heart%20failure%2C%20combined%2C%20unspec.

icd_df.loc[icd_df.icd_code.str.startswith('I50'),'icd_dx']= 'hf'
icd_df.loc[icd_df.icd_code.str.startswith('428'),'icd_dx']= 'hf'
icd_df.loc[icd_df.icd_code.str.startswith('I48'),'icd_dx']= 'af'
icd_df.loc[icd_df.icd_code.str.startswith('4273'),'icd_dx']= 'af'

icd_df_sel = icd_df.copy()[~icd_df.icd_dx.isna()]


mimic_icd_df = pd.read_csv('data/mimic-iv-3.1/hosp/diagnoses_icd.csv.gz')

dx_list = icd_df_sel.drop_duplicates(subset='icd_dx').icd_dx.to_list()

def Category_cnt(df):
    firstLoop = True
    for i in dx_list:
        df2=df.copy()
        recept_list = icd_df_sel[icd_df_sel['icd_dx']== i ]['icd_code'].to_list()
        df2[i]=df2['icd_code'].isin(recept_list).astype(int)
        df2=df2[['subject_id','hadm_id',i]].groupby(['subject_id','hadm_id']).agg({i:'max'})
        if firstLoop:
            df3=df2.copy()
            firstLoop = False
        else:
            df3 = pd.merge(df3,df2,how='left',on=['subject_id','hadm_id'])
    return df3.reset_index(drop=False)

dx_cnt = Category_cnt(mimic_icd_df) 

joblib.dump(dx_cnt,"dataset/df/dx_cnt")
