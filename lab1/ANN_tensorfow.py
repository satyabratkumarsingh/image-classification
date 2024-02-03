import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, Flatten
from  utils.image_processor import run_dlib_shape, extract_features_labels
import numpy as np

n_hidden_1 = 2048  # 1st layer number of neurons
n_hidden_2 = 2048  # 2nd layer number of neurons

def get_data():
    X, y = extract_features_labels()
  
    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:100] ; tr_Y = Y[:100]
    te_X = X[100:] ; te_Y = Y[100:]

    return tr_X, tr_Y, te_X, te_Y

def create_model():
    
    inputs = tf.keras.Input(shape=(68, 2))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(2048, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(2048, activation='sigmoid')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def train_new():
    learning_rate = 0.00001
    training_epochs = 500
    display_accuracy_step = 2

    training_images, training_labels, test_images, test_labels = get_data()

    print(training_labels)
    training_labels = np.argmax(training_labels, axis=1)
    print('=========================')
    print(training_labels)

    test_labels = np.argmax(test_labels, axis=1)

    #training_labels = np.ravel(training_labels)
    #test_labels = np.ravel(test_labels)


    print(f'Shape of training_images : {training_images.shape}')
    print(f'Shape of training_labels : {training_labels.shape}')
    print(f'Shape of test_images : {test_images.shape}')
    print(f'Shape of test_labels : {test_labels.shape}')

    
    #y1_train = training_labels[:, 0]
    #y2_train = training_labels[:, 1]

    print(training_images.shape)

    #training_labels = tf.reshape(training_labels, (-1,))
    model = create_model()

    loss_op = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=loss_op, metrics=['accuracy'])

    model.fit(training_images, training_labels, epochs=training_epochs,
              validation_data=(test_images, test_labels))
    


   