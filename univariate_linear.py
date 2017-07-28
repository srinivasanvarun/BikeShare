# clears variables -added by vishaka
all = [var for var in globals() if (var[:2], var[-2:]) != ("__", "__")]
for var in all:
    del globals()[var]

from sknn.mlp import Regressor, Layer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


import csv
test_datum = []
training_datum =[]
y_training_datum =[]
y_test_datum = []

with open('./data_set/day.csv', 'r') as days:
    days_reader = csv.DictReader(days)
    for row in days_reader:
        data = row['temp'], row['yr']
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



nn = Regressor( # Vishaka- reason for change -https://www.researchgate.net/post/why_the_prediction_or_the_output_of_neural_network_does_not_change_during_the_test_phase
    layers=[
        Layer("Linear", units = 50),
        Layer("Linear", units = 1 )],
        learning_rate=0.000005,
        n_iter=2000)


training_seasons = np.asarray(training_datum_array[:,0])
training_counts = np.asarray(y_training_datum_array[:,0])


training_seasons = np.array([[training_season] for training_season in training_seasons])
training_counts = np.array([[training_count] for training_count in training_counts])

################ normalize and mean here also- vishaka
training_seasons_mean = np.mean(training_seasons)
normalize_training_seasons = (max(training_seasons)- min(training_seasons))

training_seasons = training_seasons - training_seasons_mean/normalize_training_seasons


training_counts_mean = np.mean(training_counts)
normalize_training_counts = (max(training_counts)-min(training_counts))

training_counts = training_counts- training_counts_mean/normalize_training_counts
################ normalize and mean here also- vishaka ends

nn.fit(training_seasons, training_counts)



testing_seasons  = np.asarray(test_datum_array[:,0])
testing_counts = np.asarray(y_test_datum_array[:,0])



testing_seasons = np.array([[testing_season] for testing_season in testing_seasons])
testing_counts  = np.array([[testing_count] for testing_count in testing_counts])

################ normalize and mean here also- vishaka
normalize_testing_seasons = (max(testing_seasons)- min(testing_seasons))
mean_testing_seasons = np.mean(testing_seasons)

testing_seasons = testing_seasons - mean_testing_seasons/normalize_testing_seasons

normalize_testing_counts = (max(testing_counts)-min(testing_counts))
mean_testing_counts = np.mean(testing_counts)

testing_counts = testing_counts - mean_testing_counts/normalize_testing_counts
#################### normalize and mean here also- vishaka ends


y_output = nn.predict(testing_seasons)


##################### normalize and mean here also- vishaka
y =( testing_counts * normalize_testing_counts)+mean_testing_counts
x = ( testing_seasons * normalize_testing_seasons)+mean_testing_seasons

plt.xlabel('Temperature')
plt.ylabel('Count of Bikes Rented')
plt.suptitle('Univariate Linear Regression')

plt.plot(x, y,'ro')


y_output = (y_output*normalize_testing_counts)+np.mean(testing_counts)
plt.plot(x, y_output,'bo')

### adding legend - vishaka
red_patch = mpatches.Patch(color='red', label='Actual')
blue_patch = mpatches.Patch(color='blue', label='Predicted')
plt.legend(handles=[red_patch, blue_patch])
plt.show()
########################### normalize and mean here also- vishaka ends