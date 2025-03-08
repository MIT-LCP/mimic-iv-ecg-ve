import glob
import pandas as pd
import joblib
import os

af_list = joblib.load("dataset/df/af_list.jb")

af_list["tfrecord"] = "ecg_tf_" + af_list["ecg_pID"] +"_" + af_list["ecg_sID"] 

afdxs = glob.glob("annotation/AFclass/AFDx/*/*")
nonafdxs = glob.glob("annotation/AFclass/nonAFDx/*/*")
lowquality = glob.glob("annotation/AFclass/lowQuality/*/*")

afdxs_df = pd.DataFrame(afdxs,columns=["anno_path"])
afdxs_df["annot_Dx"] = 'AF' 
nonafdxs_df = pd.DataFrame(nonafdxs,columns=["anno_path"])
nonafdxs_df["annot_Dx"] = 'nonAF' 
lowquality_df = pd.DataFrame(lowquality,columns=["anno_path"])
lowquality_df["annot_Dx"] = 'unclassified' 

anno_df = pd.concat([afdxs_df,nonafdxs_df,lowquality_df],axis=0).reset_index(drop=True)

anno_df["tfrecord"] = anno_df["anno_path"].apply(lambda x: os.path.basename(x) ).str[:-4]

merge_df = pd.merge(af_list,anno_df,how='outer',on='tfrecord')


merge_df.annot_Dx.value_counts()

joblib.dump(merge_df,"/mnt/e/project/dataset/df/annotation_df.jb")

"""
'filename': file paths to the original ECG waveform data
'af_score': AF probability calculated by the previous reported AI model
'af': AF class by the previous reported AI model
'ecg_sID': MIMIC ECG ID
'ecg_pID': MIMIC patient ID
'tfrecord': TRrecord file name
'anno_path': ECG mages for annotations
'annot_Dx': Annotation for AF classification

AF: AF diagnosis
nonAF: non-AF diagnosis, e.g. sinus, pacemaker, ,,,
unclassified: noisy ecg or  hard to diagnosis
""""""