import os
import time
import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers, metrics
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle




save_adv_path = '../data/X_adv_FGSM.pickle'
with open(save_adv_path, 'rb') as r:
    X_train = pickle.load(r)

y_train = np.load('../data/y_test.npy') # 500
X_test = np.load('../data/X_test.npy') # 500 160 160 3
y_test = np.load('../data/y_test.npy') # 500

loss_object = losses.SparseCategoricalCrossentropy()

EPOCHS = 5
BATCH_SIZE = 2**5
MODEL_SAVE = True

base_model = tf.keras.applications.InceptionV3(input_shape=[160, 160, 3], include_top=False,weights='imagenet')
resnet_50 = tf.keras.applications.ResNet50V2(input_shape=[160, 160, 3], include_top=False, weights='imagenet')

class Model(models.Model):
    def __init__(self, base_model):
        super(Model, self).__init__()
        self.base_model = base_model
        self.top_layer = models.Sequential([
            layers.Dense(10),
            layers.Activation(tf.nn.softmax),
        ])

    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = layers.Flatten()(x)
        outputs = self.top_layer(x, training=training)
        return outputs

saved_weight_path = '../weights/model_Inception_V3_adv'
with open(saved_weight_path, 'rb') as r:
    weights = pickle.load(r)

path = '../weights/model_resnet50_V2_4_0.98_0.0730'
with open(path, 'rb') as r:
    resnet_50_weight = pickle.load(r)

path = '../weights/model_insV3_3_0.98_0.0460'
with open(path, 'rb') as r:
    inception_v3_weight = pickle.load(r)

inception_adv_model = Model(base_model)
inception_adv_model.build((None,160,160,3))
inception_adv_model.set_weights(weights)


inception_model = Model(base_model)
inception_model.build((None,160,160,3))
inception_model.set_weights(inception_v3_weight)


resnet_model = Model(resnet_50)
resnet_model.build((None,160,160,3))
resnet_model.set_weights(resnet_50_weight)




sce = losses.SparseCategoricalCrossentropy()
opt = optimizers.Adam(learning_rate=1e-4)

train_acc = metrics.SparseCategoricalAccuracy()
test_acc = metrics.SparseCategoricalAccuracy()

train_loss = metrics.Mean()
test_loss = metrics.Mean()


def train_step(inputs):

    X, y = inputs

    with tf.GradientTape() as t:
        y_pred = inception_adv_model(X)
        loss1 = sce(y, y_pred)

        y_pred = resnet_model(X)
        loss2 = sce(y, y_pred)

        y_pred = inception_model(X)
        loss3 = sce(y, y_pred)

        loss = loss1 + loss2 + loss3

    grads = t.gradient(loss, inception_adv_model.trainable_variables)
    opt.apply_gradients(list(zip(grads, inception_adv_model.trainable_variables)))

    train_acc.update_state(y, y_pred)
    train_loss.update_state(loss)


def test_step(inputs):
    X, y = inputs

    y_pred = inception_adv_model(X)
    loss = sce(y, y_pred)

    test_acc.update_state(y, y_pred)
    test_loss.update_state(loss)

BUFFER_SIZE = len(X_train)

train_dataset = tf.data.Dataset\
                .from_tensor_slices((X_train,y_train))\
                .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train_loss_list = []
train_acc_list = []

best_loss = sys.float_info.max

for e in range(EPOCHS):
    s = time.time()

    for x in train_dataset:
        train_step(x)

    if e % 1 == 0:
        print(
            f'{e + 1}/{EPOCHS}\tacc = {train_acc.result() * 100:.2f}%, loss = {train_loss.result():.8f}, {time.time() - s:.2f} sec/epoch')

    train_loss_list.append(train_loss.result())
    train_acc_list.append(train_acc.result())


    if best_loss > train_loss.result() and MODEL_SAVE:
        best_loss = train_loss.result()

        weights = inception_adv_model.get_weights()

        path = f'../weights/model_Inception_V3_adv_ens3'
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    train_loss.reset_states()
    train_acc.reset_states()

# 모델 성능 테스트
test_dataset = tf.data.Dataset\
                .from_tensor_slices((tf.squeeze(X_test),y_test))\
                .batch(BATCH_SIZE)

test_acc.reset_states()
test_loss.reset_states()

for x in test_dataset:
    test_step(x)

print(f'acc = {test_acc.result().numpy()*100:.2f}%')
print(f'loss = {test_loss.result().numpy():.4f}')

for _ in range(10):
    idx = np.random.randint(len(X_test))
    y_pred = inception_adv_model(X_test[idx:idx + 1])

    plt.imshow(X_test[idx])
    plt.title(f'real = {int(y_test[idx])}, prediction = {np.argmax(y_pred)}')
    plt.axis('off')
    plt.show()