from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from data import load_train_data, load_test_data

K.set_image_data_format('channels_last') # Tensorflow dimension ordering in this code

img_rows = 96
img_cols = 96

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def conv3x3(kernel_size,input):
    return Conv2D(kernel_size,(3,3),activation='relu',padding='same')(input)

def concatenate2x2(kernel_size,first,second):
    return concatenate([Conv2DTranspose(kernel_size, (2, 2), strides=[2, 2], padding='same')(first), second], axis=3)

def get_unet():
    inputs = Input((img_rows,img_cols,1))
    conv1 = conv3x3(32,inputs)
    conv1 = conv3x3(32,conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = conv3x3(64,pool1)
    conv2 = conv3x3(64,conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = conv3x3(128,pool2)
    conv3 = conv3x3(128,conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = conv3x3(256,pool3)
    conv4 = conv3x3(256,conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

    conv5 = conv3x3(512,pool4)
    conv5 = conv3x3(512,conv5)

    up6 = concatenate2x2(256,conv5,conv4)
    conv6 = conv3x3(256,up6)
    conv6 = conv3x3(256,conv6)

    up7 = concatenate2x2(128,conv6,conv3)
    conv7 = conv3x3(128,up7)
    conv7 = conv3x3(128,conv7)

    up8 = concatenate2x2(64,conv7,conv2)
    conv8 = conv3x3(64,up8)
    conv8 = conv3x3(64,conv8)

    up9 = concatenate2x2(32,conv8,conv1)
    conv9 = conv3x3(32,up9)
    conv9 = conv3x3(32,conv9)

    conv10= Conv2D(1,(1,1),activation='sigmoid')(conv9)
    model = Model(inputs= [inputs],outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5),loss=dice_coef,metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
    train_and_predict()
