class ArtificialNeuralNetwork:
    def __init__(self, artificial_neural_network):
        self.artificial_neural_network = artificial_neural_network

    def train_neural_network(self, X_train, y_train, batch_size=10, epochs=100):
        self.artificial_neural_network.fit(X_train, y_train, batch_size, epochs)

    #TODO
    def predict(self):
        pass



# # Fitting the ANN to the Training set
# classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
#
# # Part 3 - Making predictions and evaluating the model
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5)
#
# # Predicting a single new observation
# """Predict if the customer with the following informations will leave the bank:
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40
# Tenure: 3
# Balance: 60000
# Number of Products: 2
# Has Credit Card: Yes
# Is Active Member: Yes
# Estimated Salary: 50000"""
# new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
# new_prediction = (new_prediction > 0.5)