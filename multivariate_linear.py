all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__")]
for var in all:
    del globals()[var]


from sknn.mlp import Regressor, Layer
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.patches as mpatches
from sklearn.svm import SVR

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
    # base_array = (base_array - np.mean(base_array)) / (max(base_array) - min(base_array))
    return base_array


def destandardize(array, index):
    base_array = np.asarray(array[:, index])
    base_array = np.array([[elements] for elements in base_array])
    base_array = (base_array * (max(base_array) - min(base_array))) + np.mean(base_array)
    return base_array

def normalize(array):
    array_mean = np.mean(array)
    normalize_array_atemp = (max(array) - min(array))
    array = array - array_mean / normalize_array_atemp
    return array


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
training_counts = standardize(y_training_datum, 0)

testing_atemp = standardize(test_datum, 0)
testing_counts = standardize(y_test_datum, 0)

training_temp = normalize(training_atemp)
training_counts = normalize(training_counts)

testing_atemp = normalize(testing_atemp)
testing_counts = normalize(testing_counts)

#updated parameters - vishaka
#svr_poly = SVR(kernel='linear', degree=3, gamma='auto', coef0=0.75, tol=0.0075, C=1e10, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
svr_poly = SVR(kernel='linear', C=1e2)
# y_poly = svr_poly.fit(training_atemp, training_counts).predict(testing_atemp)
y_poly = svr_poly.fit(training_datum[:,:3], training_counts).predict(test_datum[:,:3])


plt.scatter(testing_atemp, testing_counts, color='red', label='data') # changed from plot to scatter - Vishaka
plt.hold('on')

# testing_atemp = destandardize(training_datum, 0)
# y_poly = (y_poly * (max(y_poly) - min(y_poly))) + np.mean(y_poly)

plt.scatter(testing_atemp, y_poly, color='blue', label='Polynomial model')
plt.title('Multivariate Linear')
plt.xlabel('Average Temperature')
plt.ylabel('Count of Bikes Rented')

plt.legend()
plt.show()
