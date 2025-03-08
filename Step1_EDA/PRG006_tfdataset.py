import numpy as np
import joblib, tqdm
import pandas as pd
import tensorflow as tf
import wfdb
from multiprocessing import Pool
import tqdm




def CreateTensorflowReadFile(df, out_file):
    with tf.io.TFRecordWriter(out_file) as writer:
        csv_paths = df['path_wo_ext'].to_list()
        x_ECG = np.empty((len(csv_paths), 5000, 12, 1))
       
        for i, (item_path) in enumerate(csv_paths):
            signals, _ = wfdb.rdsamp(item_path)        
            x_ECG[i, :] = signals[:, :, np.newaxis]
        
        y_sex = df['sex'].astype(float).values  # It's better to unify the type as float.
        y_age = df['age'].astype(float).values  # It's better to unify the type as float.
        y_hf = df['hf'].astype(float).values  # It's better to unify the type as float.
        y_af = df['af'].astype(float).values  # It's better to unify the type as float.
        
        # Convert data to binary file
        example = tf.train.Example(features=tf.train.Features(feature={
            "x_ECG": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_ECG.tobytes()])),
            "y_sex": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_sex.tobytes()])),
            "y_age": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_age.tobytes()])),
            "y_hf": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_hf.tobytes()])),
            "y_af": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_af.tobytes()])),
            }))

        # Write
        writer.write(example.SerializeToString())


# Load pandas DataFrame
merge_df_sel4 = joblib.load("dataset/df/merge_df_sel4")



# # To avoid running out of memory, create TensorFlow data for each record.

def process_row(i):
    # Avoid unnecessary dataframe copy
    select_df = merge_df_sel4.iloc[[i]]
    CreateTensorflowReadFile(select_df, "dataset/tfdata/tfrevised/ecg_tf_" + merge_df_sel4['ecg_pID'][i] +"_" + merge_df_sel4['ecg_sID'][i] )

if __name__ == "__main__":
    # Increasing chunk size for better parallelization
    chunk_size = 10
    with Pool(processes=8) as pool:
        list(tqdm.tqdm(pool.imap(process_row, range(len(merge_df_sel4)), chunksize=chunk_size), total=len(merge_df_sel4)))
