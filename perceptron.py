# This is the implementation of the Perceptron algorithm.
# The Perceptron algorithm can only handle linearly separable cases.

import random
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, attributes_num):
        self.attributes_num = attributes_num
        self.weight = [0 for i in range(1, attributes_num)]
        self.bias = 0

    def __skull(self, features, expected_val):
        result = self.bias
        for i in range(0, self.attributes_num - 1):
            result += features[i] * self.weight[i]
        return expected_val * result

    def __update(self, features, expected_val):
        for i in range(0, self.attributes_num - 1):
            self.weight[i] += (features[i] * expected_val)
        self.bias += expected_val

    def training(self, features_input, feature_expect, training_time = 100):
        for i in range(1, training_time):
            index = random.randint(0, len(features_input) - 1)
            if self.__skull(features_input[index], feature_expect[index]) <= 0:
                self.__update(features_input[index], feature_expect[index])

    def __show_error(self, features_input, feature_expect):
        error_num = 0
        for i in range(0, len(features_input) - 1):
            if self.__skull(features_input[i], feature_expect[i]) <= 0:
                error_num += 1
        return error_num

    def plot_accuracy(self, training_set, training_val):
        time_axis = []
        accuracy_axis = []
        for i in range(10000):
            if i % 100 == 0:
                time_axis.append(i)
                accuracy_axis.append(self.__show_error(training_set, training_val))
        plt.plot(time_axis, accuracy_axis, 'b-o', label='Accuracy over each 100 tests')
        plt.xlabel('Test time')
        plt.ylabel('Accuracy in percent')
        plt.legend()
        plt.show()

