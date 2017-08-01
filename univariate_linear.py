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





training_seasons = np.asarray(training_datum_array[:,0])
training_counts = np.asarray(y_training_datum_array[:,0])


training_seasons = np.array([[training_season] for training_season in training_seasons])
training_counts = np.array([[training_count] for training_count in training_counts])


testing_seasons  = np.asarray(test_datum_array[:,0])
testing_counts = np.asarray(y_test_datum_array[:,0])

training_seasons_mean = np.mean(training_seasons)
normalize_training_seasons = (max(training_seasons)- min(training_seasons))

training_seasons = training_seasons - training_seasons_mean/normalize_training_seasons


training_counts_mean = np.mean(training_counts)
normalize_training_counts = (max(training_counts)-min(training_counts))

training_counts = training_counts- training_counts_mean/normalize_training_counts


testing_seasons = np.array([[testing_season] for testing_season in testing_seasons])
testing_counts  = np.array([[testing_count] for testing_count in testing_counts])

normalize_testing_seasons = (max(testing_seasons)- min(testing_seasons))
mean_testing_seasons = np.mean(testing_seasons)

testing_seasons = testing_seasons - mean_testing_seasons/normalize_testing_seasons

normalize_testing_counts = (max(testing_counts)-min(testing_counts))
mean_testing_counts = np.mean(testing_counts)

testing_counts = testing_counts - mean_testing_counts/normalize_testing_counts

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', degree=3, gamma='auto', coef0=0.8, tol=0.0075, C=1e10, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

y_rbf = svr_rbf.fit(training_seasons, training_counts).predict(testing_seasons)
y_lin = svr_lin.fit(training_seasons, training_counts).predict(testing_seasons)
y_poly = svr_poly.fit(training_seasons, training_counts).predict(testing_seasons)

lw = 2

# testing_counts =( testing_counts * normalize_testing_counts)+mean_testing_counts
# testing_seasons = ( testing_seasons * normalize_testing_seasons)+mean_testing_seasons
#
plt.scatter(testing_seasons, testing_counts, color='darkorange', label='data')
plt.hold('on')
# plt.plot(testing_seasons, y_rbf, color='navy', lw=lw, label='RBF model')
# plt.plot(testing_seasons, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(testing_seasons, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

# y_output = nn.predict(testing_seasons)
#
#
# ##################### normalize and mean here also- vishaka
# y =( testing_counts * normalize_testing_counts)+mean_testing_counts
# x = ( testing_seasons * normalize_testing_seasons)+mean_testing_seasons
#
# plt.xlabel('Temperature')
# plt.ylabel('Count of Bikes Rented')
# plt.suptitle('Univariate Linear Regression')
#
# plt.plot(x, y,'ro')
#
#
# y_output = (y_output*normalize_testing_counts)+np.mean(testing_counts)
# plt.plot(x, y_output,'bo')
#
# ### adding legend - vishaka
# red_patch = mpatches.Patch(color='red', label='Actual')
# blue_patch = mpatches.Patch(color='blue', label='Predicted')
# plt.legend(handles=[red_patch, blue_patch])
# plt.show()
# ########################### normalize and mean here also- vishaka ends


# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f"
#       % np.mean((regr.predict(testing_seasons) - testing_counts) ** 2))
#
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(testing_seasons, testing_counts))
#
# # Plot outputs
# plt.scatter(testing_seasons, testing_counts,  color='black')
# plt.plot(testing_seasons, regr.predict(testing_seasons), color='blue',
#          linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()