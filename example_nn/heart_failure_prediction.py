"""
Date: 2022. 01. 12.
Programmer: MH
Description: Code for predicting death event by heart failure considering age, anaemia, creatinine_phosphokinase,
            diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time
"""
import numpy as np
import sklearn.utils
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib
import sklearn
class DeathByHeartFailurePredictor:
    """
    Class for Predicting death by features related to heart failure
    """
    def __init__(self):
        print("Matplotlib", matplotlib.__version__)
        print("TensorFlow Version", tf.__version__)
        print("Pandas Version", pd.__version__)
        print("Scikit Learn Version", sklearn.__version__)
        self.dataset = None
        self.death_predictor = None
        self.epoch = 50

    def load_dataset(self, path):
        """
        To load data from local *.CSV file
        :param path: string, Location of CSV file
        :return:
        """
        self.dataset = pd.read_csv(path)
        print(self.dataset.corr())
        pd.set_option("display.max_columns", None)
        print(self.dataset.head())
        self.dataset.info()

    def preprocess(self):
        """
        To scale whole feature's data to 0~1
        :return:
        """
        scaler = MinMaxScaler()
        result = scaler.fit_transform(self.dataset.values)
        self.dataset = pd.DataFrame(result, index=self.dataset.index, columns=self.dataset.columns)

    def split_dataset(self):
        """
        To shuffle and split dataset to training set and test set
        :return:
        """
        self.dataset = sklearn.utils.shuffle(self.dataset)
        X = self.dataset.iloc[:, :12]
        y = self.dataset.iloc[:, 12]
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, y, test_size=0.2, random_state=42)
        print("# of Data in Training Set", len(self.train_x), "# of Data in Test Set", len(self.test_x))

    def define_structure(self):
        """
        To define deep neural network structure based on fully connected layers
        :return:
        """
        # Input Layer: 12 Neurons / Activation Function: ReLU
        # Hidden Layer #1: 4 Neurons / Activation Function: ReLU
        # Output Layer: 1 Neuron  / Activation Function: Sigmoid

        self.death_predictor = tf.keras.Sequential()
        self.death_predictor.add(Dense(12, input_dim=12, activation="relu"))
        self.death_predictor.add(Dense(8, activation="relu"))
        # self.death_predictor.add(Dense(4, activation="relu"))
        self.death_predictor.add(Dense(1, activation="sigmoid"))
        self.death_predictor.summary()
        self.death_predictor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss="binary_crossentropy", metrics=["accuracy"])

    def train_model(self):
        """
        To train model
        :return:
        """
        num_train = int(len(self.train_x)*0.9)
        train_x = self.train_x.iloc[:num_train]
        train_y = self.train_y.iloc[:num_train]

        val_x = self.train_x.iloc[num_train:]
        val_y = self.train_y.iloc[num_train:]
        print(train_x.shape, train_y.shape)
        checkpoint = ModelCheckpoint(filepath="./trained_model_heart_failure/cp-{epoch:04d}.ckpt", verbose=1) # To set model checkpoint
        self.history = self.death_predictor.fit(train_x, train_y, epochs=self.epoch, batch_size=20,
                                                validation_data=(val_x, val_y), callbacks=[checkpoint], verbose=1)

    def save_model(self):
        self.death_predictor.save("./trained_model_heart_failure/model.h5") # To save model to local

    def load_model(self):
        self.death_predictor = load_model("./trained_model_heart_failure/model.h5") # To save model to local

    def predict(self, x):
        """
        To predict input data's death event
        :param x: dataframe, input data
        :return:
        """
        print(np.array(x.values).shape)

        result = self.death_predictor.predict(x.values)
        result_mod = []
        for i in range(len(result)):
            if result[i][0] >= 0.5:
                result_mod.append(1)
            else:
                result_mod.append(0)
        return result_mod

    def evaluate_model(self):
        """
        To evaluate trained model
        :return:
        """
        pred_y = self.predict(self.test_x)
        accuracy = accuracy_score(self.test_y, pred_y)
        precision = precision_score(self.test_y, pred_y)
        recall = recall_score(self.test_y, pred_y)
        result = classification_report(self.test_y, pred_y)
        print(self.test_y)
        print(pred_y)
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print(result)

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

        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    dhf = DeathByHeartFailurePredictor()
    dhf.load_dataset(r"../dataset/Heart Failure Prediction/heart_failure_clinical_records_dataset.csv")
    # dhf.load_dataset(r"../dataset/Pima Indians Diabetes Dataset/diabetes.csv")
    dhf.preprocess()
    dhf.split_dataset()
    dhf.define_structure()
    dhf.train_model()
    dhf.draw_performance_graph()
    dhf.save_model()
    dhf.load_model()
    dhf.evaluate_model()