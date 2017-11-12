from unittest import TestCase

from src.neural_network.artificial_neural_network import ArtificialNeuralNetwork
from src.neural_network.artificial_neural_network_builder import ArtificialNeuralNetworkBuilder
from src.neural_network.artificial_neural_network_options import ArtificialNeuralNetworkOptions


class ArtificialNeuralNetworkTestCase(TestCase):

    def test_can_initialise_ann_builder(self):
        builder = ArtificialNeuralNetworkBuilder()
        self.assertIsNotNone(builder)

    def test_can_return_ann_when_given_options(self):
        builder = ArtificialNeuralNetworkBuilder()
        options = ArtificialNeuralNetworkOptions()
        ann = builder.build_artificial_neural_network(options)
        self.assertIsNotNone(ann)
        self.assertIsInstance(ann, ArtificialNeuralNetwork)

    def test_returned_ann_has_been_populated(self):
        builder = ArtificialNeuralNetworkBuilder()
        options = ArtificialNeuralNetworkOptions()
        ann = builder.build_artificial_neural_network(options)
        self.assertIsNotNone(ann.artificial_neural_network)
