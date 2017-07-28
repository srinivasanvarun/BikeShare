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

def create_np_array(array, index):
    base_array = np.asarray(array[:, index])
    base_array = np.array([[elements] for elements in base_array])
    base_array = base_array - np.mean(base_array) / (max(base_array) - min(base_array))
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

# arrays = [training_datum, y_training_datum, test_datum, y_test_datum]
# print_array(arrays)

# training_datum = np.array([[elements] for elements in training_datum])

training_atemp = create_np_array(training_datum, 0)
training_hum = create_np_array(training_datum, 1)
training_ws = create_np_array(training_datum, 2)

# arrays = [training_atemp, training_hum, training_ws]
# print_array(arrays)
# print_array_shapes(arrays)

training_counts = create_np_array(y_training_datum, 0)

nn = Regressor(
    layers=[
        Layer("Sigmoid", units=100),
        Layer("Linear")],
    learning_rate=0.005,
    n_iter=3500)

nn.fit(training_datum[:,:2], y_training_datum[:,0])
# nn.fit(training_hum, training_counts)
# nn.fit(training_ws, training_counts)

testing_atemp = create_np_array(test_datum, 0)
testing_hum = create_np_array(test_datum, 1)
testing_ws = create_np_array(test_datum, 2)
testing_counts = create_np_array(y_test_datum, 0)
#
# arrays = [testing_atemp, testing_hum, testing_ws]
# print_array(arrays)
# print_array_shapes(arrays)
#
# y_output = nn.predict(testing_atemp, testing_hum, testing_ws)

y_output = nn.predict(test_datum[:,:2])

# y_output_1 = nn.predict(testing_hum)
# y_output_2 = nn.predict(testing_ws)
#
y_output =  y_output * (max(testing_counts) - min(testing_counts)) + np.mean(testing_counts)
# y_output_1 = y_output_1 * (max(testing_counts) - min(testing_counts)) + np.mean(testing_counts)
# y_output_2 = y_output_2 * (max(testing_counts) - min(testing_counts)) + np.mean(testing_counts)
#
fig, axis = plt.subplots(figsize=(3, 3))
axis.set_title('Bike Count'.format('seaborn'), color = 'green')
#
# plt.plot(testing_atemp, testing_counts, 'ro', color='blue', label ='Test data')
plt.plot(testing_atemp, y_output, 'ro', label ='Prediction')
plt.show()
#
# plt.plot(testing_hum, testing_counts, 'ro', color='blue', label = 'Test data 2')
# plt.plot(testing_hum, y_output_1, 'ro', label='Prediction')
# plt.show()
#
# plt.plot(testing_ws, testing_counts, 'ro', color='blue', label = 'Test data 3')
# plt.plot(testing_ws, y_output_2, 'ro', label='Prediction')
# plt.show()
