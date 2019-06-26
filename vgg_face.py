import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K

K.set_image_dim_ordering('tf')

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from collections import defaultdict
from glob import glob
from random import choice, sample
import math

import cv2
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import Input, Dense, BatchNormalization, Activation, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, \
    Multiply, Dropout, Subtract
from keras.models import Model
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from datetime import datetime
from keras.callbacks import TensorBoard
from keras.preprocessing import image
import keras
from keras.engine.topology import Layer
from FMLayer import FMLayer
from compact_bilinear_pooling import compact_bilinear_pooling_layer

model_name = "kinship_{}".format(datetime.now().strftime('%b%d_%H-%M-%S'))
tensor_board = TensorBoard(log_dir="../tb_log/" + model_name)

train_file_path = "input/train_relationships.csv"
train_folders_path = "input/train/"
val_famillies = "F09"

all_images = glob(train_folders_path + "*/*/*.jpg")

train_images = [x for x in all_images if val_famillies not in x]
val_images = [x for x in all_images if val_famillies in x]

train_person_to_images_map = defaultdict(list)

ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

for x in train_images:
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

val_person_to_images_map = defaultdict(list)

for x in val_images:
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

relationships = pd.read_csv(train_file_path)
relationships = list(zip(relationships.p1.values, relationships.p2.values))
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

train = [x for x in relationships if val_famillies not in x[0]]
val = [x for x in relationships if val_famillies in x[0]]


def read_img(path):
    img = image.load_img(path, target_size=(197, 197))
    # img = cv2.imread(path)
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)


def gen(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x) for x in X2])

        yield [X1, X2], labels


def baseline_model():
    input_1 = Input(shape=(197, 197, 3))
    input_2 = Input(shape=(197, 197, 3))

    # base_model = VGGFace(model='resnet50', include_top=False)
    base_model = VGGFace(model='resnet50', include_top=False)

    for x in base_model.layers[:-3]:
        x.trainable = True

    # 一、 fix architecture bug
    base_model = Model(base_model.input, base_model.layers[-2].output)

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    # x1_ = Reshape(target_shape=(7*7, 2048))(x1)
    # x2_ = Reshape(target_shape=(7*7, 2048))(x2)
    #
    # x_dot = Dot(axes=[2, 2], normalize=True)([x1_, x2_])
    # x_dot = Flatten()(x_dot)

    # x = Subtract()([x1, x2])

    # region

    # region end
    #
    # x = keras.layers.Conv2D(1024, 1)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.3)(x)

    # x = keras.layers.Conv2D(1024, 3)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.3)(x)

    #
    # x = keras.layers.Conv2D(2048, 3)(x)

    # x = GlobalAvgPool2D()(x)
    # x2 = GlobalMaxPool2D()(x)
    # x = Concatenate(axis=-1)([x1, x2])

    x1 = GlobalAvgPool2D()(x1)
    x2 = GlobalAvgPool2D()(x2)

    # x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    # x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])

    x3 = Multiply()([x3, x3])
    # x = x3
    #
    # # x = Multiply()([x1, x2])
    # # x1**2-x2**2
    x1_ = Multiply()([x1, x1])
    x2_ = Multiply()([x2, x2])
    x = Subtract()([x1_, x2_])
    # # concat
    x = Concatenate(axis=-1)([x, x3])

    # x = compact_bilinear_pooling_layer(x1.shape, x2.shape, 200, sum_pool=True)(x1, x2)

    # x = Dropout(0.1)(x)
    x = Dense(100, activation="relu")(x)
    # x = FMLayer(200, 100)(x)
    x = Dropout(0.01)(x)
    # x = Dense(50, activation="relu")(x)
    # x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

    model.summary()

    return model


file_path = "vgg_face.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)


def step_decay(epoch):
    initial_lrates = [0.00001, 0.000001, 0.0000001]

    if epoch < 20:
        return initial_lrates[0]
    elif epoch < 60:
        return initial_lrates[1]
    else:
        return initial_lrates[2]


lrate = LearningRateScheduler(step_decay)

callbacks_list = [checkpoint, lrate, tensor_board]

model = baseline_model()
# model.load_weights(file_path)
model.fit_generator(gen(train, train_person_to_images_map, batch_size=16), use_multiprocessing=True,
                    validation_data=gen(val, val_person_to_images_map, batch_size=16), epochs=100, verbose=2,
                    workers=4, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)

test_path = "input/test/"


def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


from tqdm import tqdm

submission = pd.read_csv('input/sample_submission.csv')

predictions = []

for batch in tqdm(chunker(submission.img_pair.values)):
    X1 = [x.split("-")[0] for x in batch]
    X1 = np.array([read_img(test_path + x) for x in X1])

    X2 = [x.split("-")[1] for x in batch]
    X2 = np.array([read_img(test_path + x) for x in X2])

    pred = model.predict([X1, X2]).ravel().tolist()
    predictions += pred

submission['is_related'] = predictions

submission.to_csv("vgg_face.csv", index=False)
