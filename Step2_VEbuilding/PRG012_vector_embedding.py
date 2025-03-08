import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers, Model
from tensorflow.keras import regularizers
from tqdm import tqdm
import pandas as pd
import joblib

def load_tfrecord_with_filename(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(lambda record: (record, filename))
    return dataset


def parse_with_filename(record, filename):
    features = tf.io.parse_single_example(
        record,
        features={"x_ECG": tf.io.FixedLenFeature([], tf.string),
                  "y_sex": tf.io.FixedLenFeature([], tf.string),
                  "y_age": tf.io.FixedLenFeature([], tf.string),
                  "y_hf": tf.io.FixedLenFeature([], tf.string),
                  "y_af": tf.io.FixedLenFeature([], tf.string),
                  })

    x_ECG_ = tf.io.decode_raw(features["x_ECG"], tf.float64) 
    x_ECG = tf.reshape(x_ECG_, tf.stack([5000, 12, 1]))
    return x_ECG, filename
def preprocess_with_filename(x_ECG, filename):
    start = (5000 - 4096) // 2 
    x_ECG = x_ECG[start:start+4096, :, :] 
    return x_ECG, filename


@register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)  # 必要な初期化処理を追加
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@register_keras_serializable()
class ReconstructionLossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReconstructionLossLayer, self).__init__(**kwargs)
    def call(self, inputs):
        x_true, x_pred = inputs
        loss = tf.reduce_mean(tf.square(x_true - x_pred))
        self.add_loss(loss)
        self.reconstruction_loss = loss
        return inputs

@register_keras_serializable()
class KLDivergenceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs) 
    def call(self, inputs):
        z_mean, z_log_var = inputs
        loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.add_loss(loss)
        self.kl_loss = loss
        return inputs

    
def build_vae(filter_num=32, latent_dim=128):
    inputs = layers.Input(shape=(4096, 12, 1))
    
    # Encoder
    encoding_1 = layers.Conv2D(filter_num, (5, 12), padding='same',
    kernel_regularizer=regularizers.l2(0.001))(inputs)
    encoding_1 = layers.BatchNormalization()(encoding_1)
    encoding_1 = layers.Activation('swish')(encoding_1)
    encoding_1 = layers.MaxPooling2D((4, 12), padding='same')(encoding_1)
    
    encoding_2 = layers.Conv2D(filter_num, (5, 1), padding='same',
    kernel_regularizer=regularizers.l2(0.001))(encoding_1)
    encoding_2 = layers.BatchNormalization()(encoding_2)
    encoding_2 = layers.Activation('swish')(encoding_2)
    encoding_2 = layers.MaxPooling2D((4, 1), padding='same')(encoding_2)
    
    encoding_3 = layers.Conv2D(filter_num, (5, 1), padding='same',
    kernel_regularizer=regularizers.l2(0.001))(encoding_2)
    encoding_3 = layers.BatchNormalization()(encoding_3)
    encoding_3 = layers.Activation('swish')(encoding_3)
    encoding_3 = layers.MaxPooling2D((4, 1), padding='same')(encoding_3)
    
    encoding_4 = layers.Conv2D(filter_num, (5, 1), padding='same',
    kernel_regularizer=regularizers.l2(0.001))(encoding_3)
    encoding_4 = layers.BatchNormalization()(encoding_4)
    encoding_4 = layers.Activation('swish')(encoding_4)
    encoding_4 = layers.MaxPooling2D((4, 1), padding='same')(encoding_4)

    # Latent space (mean and variance)
    encoding_f = layers.Flatten()(encoding_4)
    z_mean = layers.Dense(latent_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001))(encoding_f)
    z_log_var = layers.Dense(latent_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001))(encoding_f)
    
    # sampling layer
    z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])
    z = Sampling()([z_mean, z_log_var])
    
    # Decoder
    decoding_1 = layers.Dense(encoding_4.shape[1] * encoding_4.shape[2] * encoding_4.shape[3])(z)
    decoding_1 = layers.Reshape(encoding_4.shape[1:])(decoding_1)
    
    decoding_2 = layers.Conv2DTranspose(filter_num, (5, 1), strides=(4, 1), padding='same',
    kernel_regularizer=regularizers.l2(0.001))(decoding_1)
    decoding_2 = layers.Activation('swish')(decoding_2)
    
    decoding_3 = layers.Conv2DTranspose(filter_num, (5, 1), strides=(4, 1), padding='same',
    kernel_regularizer=regularizers.l2(0.001))(decoding_2)
    decoding_3 = layers.Activation('swish')(decoding_3)

    #  UpSampling to adjust encoding_3 size to decoding_3
    encoding_3_up = layers.UpSampling2D(size=(4, 1))(encoding_3)  
    decoding_3 = layers.Add()([decoding_3, encoding_3_up]) 
    
    decoding_4 = layers.Conv2DTranspose(filter_num, (5, 1), strides=(4, 1), padding='same',
    kernel_regularizer=regularizers.l2(0.001))(decoding_3)
    decoding_4 = layers.Activation('swish')(decoding_4)
    
    decoding_5 = layers.Conv2DTranspose(1, (4, 12), strides=(4, 12), padding='same',
    kernel_regularizer=regularizers.l2(0.001))(decoding_4)
    decoding_5 = layers.Activation('swish')(decoding_5)
    
    decoding = ReconstructionLossLayer()([inputs, decoding_5])


    vae = Model(inputs, decoding)
    encoder = Model(inputs, [z_mean, z_log_var, z])
    vae.compile(optimizer='adam', loss=None, jit_compile=False)

    # print(vae.summary())
   
    return vae, encoder

filter_num=32
latent_dim=128
batch_size = 256    
vae, encoder = build_vae(filter_num=filter_num,latent_dim=latent_dim)
encoder.load_weights('model/encoder_full_weight.weights.h5')


file_pattern = "dataset/tfdata/tftest/ecg_tf_*"
file_list = tf.data.Dataset.list_files(file_pattern)



dataset = file_list.interleave(
    load_tfrecord_with_filename,
    cycle_length=16,  
    block_length=4,  
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)





dataset = (
    dataset.map(parse_with_filename, num_parallel_calls=tf.data.experimental.AUTOTUNE)
           .map(preprocess_with_filename, num_parallel_calls=tf.data.experimental.AUTOTUNE)
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE)
)

VE_vars = ['VE_' + str(item).zfill(2) for item in range(latent_dim)]


output_csv = "embeddings/embedding_128dim.csv"

if not os.path.exists(output_csv):
    pd.DataFrame(columns=VE_vars + ['filename']).to_csv(output_csv, index=False)

for batch, filenames in tqdm(dataset):  
    predicted_value = encoder.predict(batch)
    df = pd.DataFrame(predicted_value[0], columns=VE_vars)
    df['filename'] = [f.decode('utf-8') for f in filenames.numpy()] 

    df.to_csv(output_csv, mode='a', index=False, header=False)

ve = pd.read_csv("embeddings/embedding_128dim.csv")

ve.head()


merge_df_sel4 = joblib.load("dataset/df/merge_df_sel4")
merge_df_sel4['filename'] = "dataset/tfdata/tftest/ecg_tf_" + merge_df_sel4['ecg_pID'] +"_" + merge_df_sel4['ecg_sID']

df_with_ve = pd.merge(merge_df_sel4,ve,how="inner", on=["filename"])

joblib.dump(df_with_ve, "dataset/df/test_with_ve_df")