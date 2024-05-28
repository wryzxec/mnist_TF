import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mnist_data_handler as mdh
from mnist_data_handler import MnistDataHandler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from tensorflow.keras.models import load_model

class NeuralNetwork:
    def __init__(self):
        pass

    def create_model(self):
        model = Sequential([
            Dense(40, activation = 'relu'),
            Dense(20, activation = 'relu'),
            Dense(10, activation = 'linear')
        ]
        )
        return model
    
    def compile_model(self, model, x_train, y_train):
        model.compile(
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer = tf.keras.optimizers.Adam(0.001),
        )

    def train_model_and_get_history(self, model, x_train, y_train):
        history = model.fit(
            x_train, y_train,
            epochs = 20,
        )
        return history
    
    def plot_loss(self, history):
        
        losses = history.history['loss']
        epochs = range(1, len(losses) + 1)

        plt.figure(figsize = (5, 3))
        plt.plot(epochs, losses, label = 'Training Loss')
        plt.title('Training Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
    
    def get_model_accuracy(self, predictions, y_test):
        correct_predictions = 0
        for i, prediction in enumerate(predictions): 
            predicted_num = np.argmax(prediction)
            actual_num = y_test[i]
            if (predicted_num == actual_num):
                correct_predictions += 1
        accuracy = correct_predictions/len(predictions) * 100
        return accuracy
    
    def get_incorrectly_predicted_images(self, predictions, x_test, y_test, count):
        incorrectly_predicted_images = []
        incorrectly_predicted_labels = []

        for i, prediction in enumerate(predictions): 
            predicted_num = np.argmax(prediction)
            actual_num = y_test[i]
            if (predicted_num != actual_num):
                incorrectly_predicted_images.append(x_test[i])
                incorrectly_predicted_labels.append(str(predicted_num))

        return incorrectly_predicted_images[:count], incorrectly_predicted_labels[:count]
    
def main():
    mnist_data_handler = MnistDataHandler()
    x_train, y_train = mnist_data_handler.load_training_data()
    x_test, y_test = mnist_data_handler.load_test_data()

    neural_network = NeuralNetwork()
    model = neural_network.create_model()

    neural_network.compile_model(model, x_train, y_train)
    history = neural_network.train_model_and_get_history(model, x_train, y_train)

    neural_network.plot_loss(history)

    prediction = model.predict(np.reshape(x_test[1], (1, 28*28)))
    print(np.argmax(prediction))

    predictions = model.predict(x_test)
    
    accuracy = neural_network.get_model_accuracy(predictions, y_test)
    print('Accuracy: ', accuracy)

    incorrectly_predicted_images, incorrectly_predicted_labels = neural_network.get_incorrectly_predicted_images(predictions, x_test, y_test, 10)
    mnist_data_handler.show_images(incorrectly_predicted_images[:10], incorrectly_predicted_labels[:10])
            

if __name__ == "__main__":
    main()