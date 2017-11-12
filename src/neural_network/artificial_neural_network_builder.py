
from src.neural_network.artificial_neural_network import ArtificialNeuralNetwork
import keras
from keras.models import Sequential
from keras.layers import Dense


class ArtificialNeuralNetworkBuilder:
    def build_artificial_neural_network(self, options):
        classifier = self._assemble_classifier(options)
        classifier.compile(optimizer=options.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return ArtificialNeuralNetwork(artificial_neural_network=classifier)

    def _assemble_classifier(options):
        classifier = Sequential()
        for index, layer in enumerate(options.layers_description):
            if (index == 0):
                classifier.add(Dense(units=layer.units, kernel_initializer='uniform', activation=layer.activation,
                                     input_dim=options.input_dimension))
            else:
                classifier.add(Dense(units=layer.units, kernel_initializer='uniform', activation=layer.activation))
        return classifier



