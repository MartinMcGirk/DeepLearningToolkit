from src.neural_network.layer_description import LayerDescription


class ArtificialNeuralNetworkOptions:
    def __init__(self, input_dimension, layers_description=None, optimizer='adam'):
        self.input_dimension = input_dimension
        self.layers_description = layers_description or self._getDefaultLayerDescription()
        self.optimizer = optimizer

    def _getDefaultLayerDescription(self, input_dimension):
        default_layer_size = input_dimension // 2
        return [
            LayerDescription(units=default_layer_size, activation='relu'),
            LayerDescription(units=default_layer_size, activation='relu'),
            LayerDescription(units=1, activation='sigmoid')
        ]