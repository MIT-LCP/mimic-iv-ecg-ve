import os
import tensorflow as tf
import shutil


source_dir = "dataset/tfdata/tfrevised"  
test_dir = "dataset/tfdata/tftest"   
train_dir = "dataset/tfdata/tftrain"   

os.makedirs(test_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)


def copy_files():
    files = tf.io.gfile.listdir(source_dir)

    for file in files:
        source_path = os.path.join(source_dir, file)

        if any(f"p{str(i)}" in file for i in range(1800, 2000)):
            target_path = os.path.join(test_dir, file)
        else:
            target_path = os.path.join(train_dir, file)

        tf.io.gfile.copy(source_path, target_path, overwrite=True)

copy_files()
