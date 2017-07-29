all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__")]
for var in all:
    del globals()[var]


from sknn.mlp import Regressor, Layer
import numpy as np
import matplotlib.pyplot as plt
import csv

def print_array(arrays):
    for array in arrays:
        print('\n\n')
        print(array)
        print(len(array))

def print_array_shapes(arrays):
    for array in arrays:
        print('\n\n')
        print(array.shape)

def standardize(array, index):
    base_array = np.asarray(array[:, index])
    base_array = np.array([[elements] for elements in base_array])
    base_array = (base_array - np.mean(base_array)) / (max(base_array) - min(base_array))
    return base_array


def destandardize(array, index):
    base_array = np.asarray(array[:, index])
    base_array = np.array([[elements] for elements in base_array])
    base_array = (base_array * (max(base_array) - min(base_array))) + np.mean(base_array)
    return base_array

def parse_data():
    with open('./data_set/day.csv', 'r') as days:

        test_datum = []
        training_datum = []
        y_training_datum = []
        y_test_datum = []

        days_reader = csv.DictReader(days)

        for row in days_reader:
            data = row['atemp'], row['hum'], row['windspeed'], row['yr']
            count_data = row['cnt'],row['yr']

            if data[3] == '0':
                training_datum.append(data)

            else:
                test_datum.append(data)

            if count_data[1] == '0':
                y_training_datum.append(count_data)
            else:
                y_test_datum.append(count_data)

        training_datum_array = np.asarray(training_datum, dtype='float')
        y_training_datum_array = np.asarray(y_training_datum, dtype='float')
        test_datum_array = np.asarray(test_datum, dtype='float')
        y_test_datum_array = np.asarray(y_test_datum, dtype='float')

        return [training_datum_array, y_training_datum_array, test_datum_array, y_test_datum_array]


training_datum, y_training_datum, test_datum, y_test_datum = parse_data()

training_atemp = standardize(training_datum, 0)
training_hum = standardize(training_datum, 1)
training_ws = standardize(training_datum, 2)
training_counts = standardize(y_training_datum, 0)

nn = Regressor(
    layers=[
        Layer("Sigmoid", units = 50),
        Layer("Linear", units = 5),
        Layer("Linear", units = 1)],
    learning_rate=0.000005,
    n_iter=2000)


# nn = Regressor( # Vishaka- reason for change -https://www.researchgate.net/post/why_the_prediction_or_the_output_of_neural_network_does_not_change_during_the_test_phase
#     layers=[
#         Layer("Sigmoid", units=50), #28 was optimum,
#         Layer("Linear", units = 5),#5
#        # Layer("Linear", units = 4),#5 was optimum
#         Layer("Linear", units = 1 )],
#     learning_rate=0.0000005,
#     n_iter=2000)

nn.fit(training_datum[:,:3], y_training_datum[:,0])

testing_atemp = standardize(test_datum, 0)
testing_hum = standardize(test_datum, 1)
testing_ws = standardize(test_datum, 2)
testing_counts = standardize(y_test_datum, 0)

y_output = nn.predict(test_datum[:,:3])

predicted_output = destandardize(y_output, 0)
actual_ouput = destandardize(y_test_datum, 0)

testing_atemp = destandardize(testing_atemp, 0)

# fig, axis = plt.subplots(figsize=(6, 6))
# axis.set_title('Bike Count'.format('seaborn'), color = 'green')
#
# plt.plot(testing_atemp, testing_counts, 'ro', color='blue', label ='Test data')

plt.xlabel('Average Temperature')
plt.ylabel('Count of Bikes Rented')
plt.suptitle('Multivariate Non Linear Regression')

plt.plot(testing_atemp, actual_ouput, 'ro')
plt.plot(testing_atemp, predicted_output, 'bo')
plt.show()
