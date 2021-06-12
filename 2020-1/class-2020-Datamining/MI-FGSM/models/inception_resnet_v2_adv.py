import os
import time
import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers, metrics
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle



y_train = np.load('../data/y_test.npy') # 500
X_test = np.load('../data/X_test.npy') # 500 160 160 3
y_test = np.load('../data/y_test.npy') # 500

save_adv_path = '../data/X_adv_FGSM.pickle'
with open(save_adv_path, 'rb') as r:
    X_train = pickle.load(r)

print(np.shape(X_train))
loss_object = losses.SparseCategoricalCrossentropy()

EPOCHS = 5
BATCH_SIZE = 2**5
MODEL_SAVE = True

base_model = tf.keras.applications.InceptionResNetV2(input_shape=[160, 160, 3], include_top=False,weights='imagenet')


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

model = Model(base_model)

sce = losses.SparseCategoricalCrossentropy()
opt = optimizers.Adam(learning_rate=1e-4)

train_acc = metrics.SparseCategoricalAccuracy()
test_acc = metrics.SparseCategoricalAccuracy()

train_loss = metrics.Mean()
test_loss = metrics.Mean()


def train_step(inputs):
    X, y = inputs

    with tf.GradientTape() as t:
        y_pred = model(X)
        loss = sce(y, y_pred)

    grads = t.gradient(loss, model.trainable_variables)
    opt.apply_gradients(list(zip(grads, model.trainable_variables)))

    train_acc.update_state(y, y_pred)
    train_loss.update_state(loss)


def test_step(inputs):
    X, y = inputs

    y_pred = model(X)
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

        weights = model.get_weights()

        path = f'../weights/model_Inception_res_adv'
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
    y_pred = model(X_test[idx:idx + 1])

    plt.imshow(X_test[idx])
    plt.title(f'real = {int(y_test[idx])}, prediction = {np.argmax(y_pred)}')
    plt.axis('off')
    plt.show()