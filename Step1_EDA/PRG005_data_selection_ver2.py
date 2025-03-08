# load library
import numpy as np
import pandas as pd
import wfdb
import joblib
import glob
import tqdm
from IPython.display import display
from sklearn.model_selection import train_test_split

merge_df_sel3 = joblib.load("dataset/df/merge_df_sel3")
merge_df_sel4 = merge_df_sel3.copy()[~merge_df_sel3.del_ecg]


merge_df_sel4.loc[merge_df_sel4.subject_id<18000000,'total_set'] = 'train'
merge_df_sel4.loc[merge_df_sel4.subject_id>=18000000,'total_set'] = 'test'

merge_df_sel4.loc[(merge_df_sel4['total_set']=='test'),'ds50K'] = 'test' 
merge_df_sel4.loc[(merge_df_sel4['total_set']=='test'),'ds_m_50K'] = 'test' 
merge_df_sel4.loc[(merge_df_sel4['total_set']=='test'),'ds_w_50K'] = 'test' 

merge_df_sel4.loc[(merge_df_sel4[ merge_df_sel4['total_set'] != 'test'].sample(50000,random_state=0).index), 'ds50K']= 'train' 
merge_df_sel4.loc[(merge_df_sel4[ (merge_df_sel4['total_set'] != 'test') & (merge_df_sel4['gender']=="M")].sample(50000,random_state=0).index), 'ds_m_50K']= 'train' 
merge_df_sel4.loc[(merge_df_sel4[ (merge_df_sel4['total_set'] != 'test') & (merge_df_sel4['race2']=="White")].sample(50000,random_state=0).index), 'ds_w_50K']= 'train' 

merge_df_sel4['ecg_id'] = "ecg_" + merge_df_sel4['total_set'].replace(
    {'train': '1', 'test': '2'}).apply(lambda x: x if x in ["1", "2"] else '0') + \
    merge_df_sel4['ds50K'].replace(
    {'train': '1', 'test': '2'}).apply(lambda x: x if x in ["1", "2"] else '0') + \
    merge_df_sel4['ds_m_50K'].replace(
    {'train': '1', 'test': '2'}).apply(lambda x: x if x in ["1", "2"] else '0') + \
    merge_df_sel4['ds_w_50K'].replace(
    {'train': '1', 'test': '2'}).apply(lambda x: x if x in ["1", "2"] else '0') + \
    "_" + merge_df_sel4.index.astype(str).str.zfill(8)

merge_df_sel4['ecg_sID']= merge_df_sel4['path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
merge_df_sel4['ecg_pID']= merge_df_sel4['path'].apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))))

merge_df_sel4 = merge_df_sel4.reset_index(drop=True)

joblib.dump(merge_df_sel4,"dataset/df/merge_df_sel4")

display(merge_df_sel4['total_set'].value_counts())
display(merge_df_sel4['ds50K'].value_counts())
display(merge_df_sel4['ds_m_50K'].value_counts())
display(merge_df_sel4['ds_w_50K'].value_counts())


# train    328895
# test     143686
# Name: total_set, dtype: int64


# test     143686
# train     50000
# Name: ds50K, dtype: int64