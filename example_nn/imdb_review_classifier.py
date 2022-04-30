"""
Date: 2020. 09. 10.
Programmer: MH
Description: Code for classifying IMDB dataset applying fully connected neural networks
"""
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import load_model
import os

tf.keras.backend.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


class IMDBReviewClassifier:
    def __init__(self):
        self.vocab_size = 10000

    def load_dataset(self):
        """
        To load dataset from local
        :return:
        """
        self.data = keras.datasets.imdb
        (self.train_X, self.train_y), (self.test_X, self.test_y) = self.data.load_data(num_words=10000) # To select 10,000 words emerging frequently
        print("# of Instances in Training Set: ", len(self.train_X))
        print("# of Instances in Test Set: ", len(self.test_X))

        self.word_index = self.data.get_word_index()    # To load word index
        self.word_index = {k: (v + 3) for k, v in self.word_index.items()}  # To set word index
        # To define some words
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNK>"] = 2  # unknown
        self.word_index["<UNUSED>"] = 3
        self.decode_word_index = dict([(v, k) for (k, v) in self.word_index.items()])  # To change location of key and value in word index
        print(self.train_X[0])
        print(self.decode_review(self.train_X[0]))

    def decode_review(self, text):
        """
        To decode
        :param text: sting, target text
        :return: string, decoded text
        """
        return " ".join([self.decode_word_index.get(i, "?") for i in text])    # To return decoded text

    def preprocess_dataset(self):
        """
        To add padding at data for make whole text as same length
        :return:
        """
        self.train_X = keras.preprocessing.sequence.pad_sequences(self.train_X,
                                                        value=self.word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

        self.test_X = keras.preprocessing.sequence.pad_sequences(self.test_X,
                                                               value=self.word_index["<PAD>"],
                                                               padding='post',
                                                               maxlen=256)
        print(self.train_X[0])
        print(self.decode_review(self.train_X[0]))


    def define_structure(self):
        """
        To define neural network structure
        :return:
        """
        self.model = keras.Sequential()
        self.model.add(keras.layers.Embedding(self.vocab_size, 16, input_shape=(None,)))
        self.model.add(keras.layers.GlobalAveragePooling1D())
        self.model.add(keras.layers.Dense(1000, activation='relu'))
        self.model.add(keras.layers.Dense(100, activation='relu'))
        self.model.add(keras.layers.Dense(16, activation='relu'))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

        self.model.summary()
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def train_model(self):
        """
        To train model
        :return:
        """
        x_val = self.train_X[:10000]    # Features in validation Set
        x_train = self.train_X[10000:]  # Features in Training Set

        y_val = self.train_y[:10000]    # Features in validation Set
        y_train = self.train_y[10000:]  # Features in Training Set

        checkpoint = ModelCheckpoint(filepath="./trained_model/cp-{epoch:04d}.ckpt", verbose=1) # To set model checkpoint
        self.history = self.model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val),
                                      callbacks=[checkpoint], verbose=1)
        self.model.save("./trained_model/IMDBReviewClassifier/model.h5") # To save model to local

    def predict(self, x):
        """
        To predict input data
        :param x: dataframe, input data
        :return:
        """
        self.model = load_model("./trained_model/IMDBReviewClassifier/model.h5") # To load model from local
        result = self.model.predict(x)
        return result

    def evaluate_model(self,):
        """
        To evaluate model
        :param y_true: list, true labels
        :param y_pred: list, predicted labels
        :return:
        """
        result = self.model.evaluate(self.test_X, self.test_y, verbose=2)
        return {"Loss": result[0], "Accuracy": result[1]}

    def draw_performance_graph(self):
        """
        To draw loss graph
        :return:
        """
        history_dict = self.history.history
        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    fc = IMDBReviewClassifier()
    fc.load_dataset()
    fc.preprocess_dataset()
    fc.define_structure()
    fc.train_model()
    result = fc.evaluate_model()
    print("Accuracy: ", result["Accuracy"])
    print("Loss: ", result["Loss"])
    fc.draw_performance_graph()
