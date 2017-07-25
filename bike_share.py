from sknn.mlp import Regressor, Layer
import numpy as np
import matplotlib.pyplot as plt

import csv

test_datum = [] #[[] for _ in range(400)]
training_datum =[]# [[] for _ in range(400)]
y_training_datum =[]# [[] for _ in range(400)]
y_test_datum = []#[[] for _ in range(400)]

with open('./data_set/day.csv', 'r') as days:
    days_reader = csv.DictReader(days)
    for row in days_reader:
        data = row['season'], row['yr']
        count_data =  row['cnt'],row['yr']

        if data[1] == '0':
            training_datum.append(data)
            #training_datum = np.append(data[0])#data[0]# np.concatenate((training_datum, data[0]),axis = 0)

        else:
            #test_datum = data[0]#np.concatenate((test_datum, data[0]), axis=0)
            test_datum.append(data)

        if count_data[1] == '0':
            #y_training_datum = count_data[0]#np.concatenate((y_training_datum, count_data[0]), axis=0)
            y_training_datum.append(count_data)
        else:
            #y_test_datum = count_data[0]# np.concatenate((y_test_datum, count_data[0]),axis=0)
            y_test_datum.append(count_data)


training_datum_array = np.asarray(training_datum)
training_datum_array = training_datum_array.astype(np.float)
y_training_datum_array = np.asarray(y_training_datum)
y_training_datum_array = y_training_datum_array.astype(np.float)

# print(training_datum_array.shape)
# print(test_datum_array.shape)
# print(training_datum_array)
# print(test_datum_array)

# print(test_datum)

test_datum_array = np.asarray(test_datum)
test_datum_array = test_datum_array.astype(np.float)
y_test_datum_array = np.asarray(y_test_datum)
y_test_datum_array = y_test_datum_array.astype(np.float)


nn = Regressor(layers=[Layer("Rectifier", units=100), Layer("Linear")]
                , warning=None, parameters=None, random_state=None, learning_rule=u'sgd', learning_rate=0.5,
               learning_momentum=0.9, normalize=None, regularize=None, weight_decay=None, dropout_rate=None,
               batch_size=1, n_iter=None, n_stable=10, f_stable=0.001, valid_set=None, valid_size=0.0, loss_type=None,
               callback=None, debug=False, verbose=None)

nn.fit(training_datum_array, y_training_datum_array)


y_output = nn.predict(test_datum_array)
print(y_output)

print("\n\n")

print(y_test_datum_array)

# plt.plot(test_datum_array[0], y_output[0])
# plt.show();
# plt.plot(test_datum_array[0], y_test_datum_array[0])
# plt.show();
