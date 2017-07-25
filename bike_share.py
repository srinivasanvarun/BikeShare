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
        data = row['temp'], row['yr']
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

#print(training_datum_array)
# print(test_datum_array.shape)
# print(training_datum_array)
# print(test_datum_array)

# print(test_datum)

test_datum_array = np.asarray(test_datum)
test_datum_array = test_datum_array.astype(np.float)
y_test_datum_array = np.asarray(y_test_datum)
y_test_datum_array = y_test_datum_array.astype(np.float)

# print("dzgsgs")
# print(training_datum_array[:,0])

nn = Regressor(
    layers=[
        Layer("Sigmoid", units=100),
        Layer("Linear")],
    learning_rate=0.001,
    n_iter=2000)

# test_datum_array = test_datum_array[test_datum_array[:, 1].argsort()]
# y_test_datum_array = y_test_datum_array[y_test_datum_array[:,1].argsort()]

training_seasons = np.asarray(training_datum_array[:,0])
training_counts = np.asarray(y_training_datum_array[:,0])


training_seasons = np.array([[training_season] for training_season in training_seasons])
training_counts = np.array([[training_count] for training_count in training_counts])


training_seasons = training_seasons - np.mean(training_seasons)/(max(training_seasons)- min(training_seasons))
training_counts = training_counts- np.mean(training_counts)/(max(training_counts)-min(training_counts))

print(training_seasons)
nn.fit(training_seasons, training_counts)


# test_datum_array = test_datum_array[test_datum_array[:, 1].argsort()]
# y_test_datum_array = y_test_datum_array[y_test_datum_array[:,1].argsort()]

testing_seasons  = np.asarray(test_datum_array[:,0])
testing_counts = np.asarray(y_test_datum_array[:,0])



testing_seasons = np.array([[testing_season] for testing_season in testing_seasons])
testing_counts  = np.array([[testing_count] for testing_count in testing_counts])

testing_seasons = testing_seasons - np.mean(testing_seasons)/(max(testing_seasons)- min(testing_seasons))
testing_counts = testing_counts - np.mean(testing_counts)/(max(testing_counts)-min(testing_counts))



y_output = nn.predict(testing_seasons)
print(y_output)
#
print("\n\n")
#
print(testing_counts)

plt.plot(testing_seasons, testing_counts,'ro')
plt.show()
y_output = y_output*(max(testing_counts)-min(testing_counts))+np.mean(testing_counts)
plt.plot(testing_seasons, y_output,'ro')
plt.show()
