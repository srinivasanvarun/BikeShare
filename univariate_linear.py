# clears variables -added by vishaka
all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__")]
for var in all:
    del globals()[var]

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

import csv
test_datum = []
training_datum =[]
y_training_datum =[]
y_test_datum = []

with open('./data_set/day.csv', 'r') as days:
    days_reader = csv.DictReader(days)
    for row in days_reader:
        data = row['atemp'], row['yr']
        count_data =  row['cnt'],row['yr']

        if data[1] == '0':
            training_datum.append(data)


        else:
            test_datum.append(data)

        if count_data[1] == '0':

            y_training_datum.append(count_data)
        else:

            y_test_datum.append(count_data)


training_datum_array = np.asarray(training_datum)
training_datum_array = training_datum_array.astype(np.float)
y_training_datum_array = np.asarray(y_training_datum)
y_training_datum_array = y_training_datum_array.astype(np.float)



test_datum_array = np.asarray(test_datum)
test_datum_array = test_datum_array.astype(np.float)
y_test_datum_array = np.asarray(y_test_datum)
y_test_datum_array = y_test_datum_array.astype(np.float)





training_atemp = np.asarray(training_datum_array[:, 0])
training_counts = np.asarray(y_training_datum_array[:,0])


training_atemp = np.array([[training_season] for training_season in training_atemp])
training_counts = np.array([[training_count] for training_count in training_counts])


testing_atemp  = np.asarray(test_datum_array[:, 0])
testing_counts = np.asarray(y_test_datum_array[:,0])

training_atemp_mean = np.mean(training_atemp)
normalize_training_atemp = (max(training_atemp) - min(training_atemp))

training_atemp = training_atemp - training_atemp_mean / normalize_training_atemp

training_counts_mean = np.mean(training_counts)
normalize_training_counts = (max(training_counts)-min(training_counts))

training_counts = training_counts- training_counts_mean/normalize_training_counts


testing_atemp = np.array([[testing_season] for testing_season in testing_atemp])
testing_counts  = np.array([[testing_count] for testing_count in testing_counts])

normalize_testing_atemp = (max(testing_atemp) - min(testing_atemp))
mean_testing_atemp = np.mean(testing_atemp)

testing_atemp = testing_atemp - mean_testing_atemp / normalize_testing_atemp

normalize_testing_counts = (max(testing_counts)-min(testing_counts))
mean_testing_counts = np.mean(testing_counts)

testing_counts = testing_counts - mean_testing_counts/normalize_testing_counts

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', degree=3, gamma='auto', coef0=0.8, tol=0.0075, C=1e10, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

y_rbf = svr_rbf.fit(training_atemp, training_counts).predict(testing_atemp)
y_lin = svr_lin.fit(training_atemp, training_counts).predict(testing_atemp)
y_poly = svr_poly.fit(training_atemp, training_counts).predict(testing_atemp)

lw = 2

# testing_counts =( testing_counts * normalize_testing_counts)+mean_testing_counts
# testing_seasons = ( testing_seasons * normalize_testing_seasons)+mean_testing_seasons
#
plt.scatter(testing_atemp, testing_counts, color='red', label='data')
plt.hold('on')
# plt.plot(testing_seasons, y_rbf, color='navy', lw=lw, label='RBF model')
# plt.plot(testing_seasons, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(testing_atemp, y_poly, color='blue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

