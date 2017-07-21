from sknn.mlp import Regressor, Layer
import numpy as np
import matplotlib.pyplot as plt

import csv

with open('./data_set/day.csv', 'r') as days:
    days_reader = csv.DictReader(days)
    for row in days_reader:
       data = row['instant'], row['dteday'], row['season'], row['atemp']


# X_train = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
# X_test = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
# y_train = np.array([[1., 0.], [1., 1.], [1., 0.], [1., 1.]])
# y_test = np.array([[1., 0.], [1., 1.], [1., 0.], [1., 1.]])
#
# nn = Regressor(layers=[Layer("Rectifier", units=100), Layer("Linear")]
#                , warning=None, parameters=None, random_state=None, learning_rule=u'sgd', learning_rate=0.5,
#                learning_momentum=0.9, normalize=None, regularize=None, weight_decay=None, dropout_rate=None,
#                batch_size=1, n_iter=None, n_stable=10, f_stable=0.001, valid_set=None, valid_size=0.0, loss_type=None,
#                callback=None, debug=False, verbose=None)
#
# nn.fit(X_train, y_train)
# y_output = nn.predict(X_test)
# print(y_output)
# plt.plot(X_test, y_output)
# plt.show();
# plt.plot(X_test, y_test)
# plt.show();
