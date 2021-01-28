import tensorflow as tf
import tensorflow.keras.backend as K

def lad_loss(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))

def ssim_metric(trueY, predY):
    return tf.image.ssim(trueY, predY, max_val=1.)

def psnr_metric(trueY, predY):
    return tf.image.psnr(trueY, predY, max_val=1.)