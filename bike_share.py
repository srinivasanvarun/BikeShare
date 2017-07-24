from sknn.mlp import Regressor, Layer
import numpy as np
import matplotlib.pyplot as plt

import csv

test_datum = []
training_datum = np.array([])
y_training_datum = np.array([])
y_test_datum = np.array([])

with open('./data_set/day.csv', 'r') as days:
    days_reader = csv.DictReader(days)
    for row in days_reader:
        data = row['instant'], row['dteday'], row['season'], row['atemp'], row['yr']
        count_data = row['yr'], row['cnt']

        if data[4] == '0':
            #training_datum.append(data*5)
            #training_datum = np.concatenate((training_datum, data),axis = 0)

        else:
            test_datum = np.concatenate((test_datum, data), axis=0)

        if count_data[0] == '0':
            y_training_datum = np.concatenate((y_training_datum, count_data), axis=0)
        else:
            y_test_datum = np.concatenate((y_test_datum, count_data),axis=0)

final_train_data = training_datum[1]

print(y_training_datum)

nn = Regressor(layers=[Layer("Rectifier", units=100), Layer("Linear")]
                , warning=None, parameters=None, random_state=None, learning_rule=u'sgd', learning_rate=0.5,
               learning_momentum=0.9, normalize=None, regularize=None, weight_decay=None, dropout_rate=None,
               batch_size=1, n_iter=None, n_stable=10, f_stable=0.001, valid_set=None, valid_size=0.0, loss_type=None,
               callback=None, debug=False, verbose=None)

# nn.fit(training_datum, y_training_datum)
# y_output = nn.predict(training_datum)
# print(y_output)
#plt.plot(X_test, y_output)
#plt.show();
#plt.plot(X_test, y_test)
#plt.show();
