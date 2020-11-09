# Artificial Neural Network

# Importing the libraries
# https://www.tandfonline.com/doi/abs/10.1080/10106049.2015.1094522?journalCode=tgei20&
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout #for overfiting problem (stay in p rang 0.1-0.5)
import keras.backend as K
import tensorflow as tf

from GEOUtil_CommonUtils import GetFilePath
file_path = GetFilePath() + '3_CreatingDataset/'

#lossF = 'mse'
lossF = 'mae'
#pollutant = 'SO2'
#pollutant = 'NOX as NO2'
#pollutant = 'O3'
#pollutant = 'NO'
#pollutant = 'PM10'
pollutant = 'NO2'
epochsNo = 10000

# Importing the dataset
dataset = pd.read_csv(file_path + 'data_set_a.csv', sep = ';')
iloscP = 10
for band in range(1, 12, 1):
    #IN = (dataset['B{}_{}'.format(band, 1)].values).astype(np.float32)
    IN = 0
    for i in range(1,iloscP + 1):
        IN += (dataset['B{}_{}'.format(band, i)].values).astype(np.float32)
    IN = IN / iloscP
    if (band < 10):
        # REFLECTANCE TOA
        MR = (dataset['REFLECTANCE_MULT_BAND_{}'.format(band)]).astype(np.float32)
        AR =  (dataset['REFLECTANCE_ADD_BAND_{}'.format(band)]).astype(np.float32)
        SE = (dataset['SUN_ELEVATION']).astype(np.float32)
        dataset['B{}_av'.format(band)] = (MR * IN + AR) / np.sin(np.deg2rad(SE))
    else:
        # RADIANCE TOA
        K1 = (dataset['K1_CONSTANT_BAND_{}'.format(band)]).astype(np.float32)
        K2 = (dataset['K2_CONSTANT_BAND_{}'.format(band)]).astype(np.float32)
        RA = (dataset['RADIANCE_ADD_BAND_{}'.format(band)]).astype(np.float32)
        RM = (dataset['RADIANCE_MULT_BAND_{}'.format(band)]).astype(np.float32)
        dataset['B{}_av'.format(band)] =  K2 / np.log(K1 / (IN*RM +RA) + 1)
    #dataset['B{}_av'.format(band)] = IN
dataset.to_csv(file_path + 'data_set_details.csv', index = False, header=True, sep = ';')

dataset = dataset[dataset[pollutant] > 0]
dataset.drop(['QA_terrain', 'QA_radiometric'], axis=1, inplace=True)
#print(dataset, '\n')

from GEOUtil_CommonUtils import bandsMinMaxScaller
bands = ['QA_fill',	'QA_cloudConf', 'QA_cloud',	'QA_cloudShadow', 'QA_snowIce', 'QA_cirrusConf', 
         'B1_av', 'B2_av', 'B3_av',	'B4_av', 'B5_av', 'B6_av', 'B7_av', 'B8_av', 'B9_av', 'B10_av', 'B11_av']
for band in bands:
   dataset[band] = bandsMinMaxScaller(band, dataset[band])
X_scaled = dataset.loc[:,'QA_fill':'B11_av'].values
print(X_scaled, '\n\n', X_scaled.shape, '\n\n')

dataset[pollutant] = bandsMinMaxScaller(pollutant, dataset[pollutant])
Y_scaled = dataset.loc[:, pollutant].values
print(Y_scaled, '\n', Y_scaled.shape, '\n\n')

# Feature Scaling (can be removed if we want to plot features)
#X = dataset.loc[:,'QA_fill':'B11_av'].values
#print(X, '\n\n', X.shape, '\n\n')
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
#scalarX = MinMaxScaler()
#scalarX = StandardScaler()
#X_scaled = scalarX.fit_transform(X)

# Feature Scaling (can be removed if we want to plot features)
#Y = dataset.loc[:, pollutant].values
#print(Y, '\n', Y.shape, '\n\n')
#scalarY = MinMaxScaler()
#Y_scaled = Y.reshape(-1, 1)
#Y_scaled = scalarY.fit_transform(Y_scaled)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size = 0.2, random_state = 0)

df = pd.DataFrame(X_train, columns=dataset.loc[:,'QA_fill':'B11_av'].columns)
df['Y'] = Y_train
df.to_csv(file_path + 'train.csv', index = False, header=True, sep = ';')

import matplotlib.pyplot as plt
def plt_hist():
    fig, ax = plt.subplots(1, 3)
    ax[0].hist(X_train[:, 0:0], 10, facecolor='red', alpha=0.5, label="band 10")
    ax[1].hist(X_train[:, 1:1], 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="band 11")
    ax[2].hist(Y_train, 10, facecolor='blue', ec="black", lw=0.5, alpha=0.5, label="co2")
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
    fig.suptitle("band 10 i 11")
    ax[0].set_ylim([0, 90])
    ax[1].set_ylim([0, 90])
    ax[0].set_xlabel("Band 10")
    ax[0].set_ylabel("Value")
    ax[1].set_xlabel("Band 11")
    ax[1].set_ylabel("Value")
    plt.show()
#plt_hist()

def plt_points():
    plt.subplot(3, 1, 1)
    plt.plot(X_train[:, 0], Y_train, 'ro')
    plt.title('Band 10')
    plt.ylabel('Co2')
    plt.subplot(3, 1, 2)
    plt.plot(X_train[:, 5], Y_train, 'ro')
    plt.title('Band 11')
    plt.ylabel('Co2')
    plt.subplot(3, 1, 3)
    plt.plot(X_train[:, 7], Y_train, 'ro')
    plt.title('Band 11')
    plt.ylabel('Co2')
    plt.show()
#plt_points()

#Making ANN
#Initialising the ANN
# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
# http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/
# https://towardsdatascience.com/a-gentle-journey-from-linear-regression-to-neural-networks-68881590760e
classifier = Sequential()

# Adding the input layer and the first hiden layer
# https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/
# https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/
#classifier.add(Dense(11, activation='relu', input_shape=(11,)))
#classifier.add(Dropout(p = 0.1))
#classifier.add(Dense(25, input_dim=11, activation='relu', kernel_initializer='he_uniform'))
classifier.add(Dense(55, input_dim=17, activation='relu'))
#classifier.add(Dropout(p = 0.1))
classifier.add(Dense(55, activation='relu'))
#classifier.add(Dropout(p = 0.1))
classifier.add(Dense(55, activation='relu'))
#classifier.add(Dropout(p = 0.1))
classifier.add(Dense(55, activation='relu'))
#classifier.add(Dropout(p = 0.1))
classifier.add(Dense(55, activation='relu'))
#classifier.add(Dropout(p = 0.1))
classifier.add(Dense(55, activation='relu'))
#classifier.add(Dropout(p = 0.1))
classifier.add(Dense(1, activation='linear'))
#classifier.add(Dense(1, activation='sigmoid'))

# Compiling the ANN
# Regression Loss Functions All Machine Learners Should Know
# https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
# Measuring the accurancy for regresion model
# https://stackoverflow.com/questions/50797135/why-does-this-neural-network-have-zero-accuracy-and-very-low-loss
from GEOUtil_CommonUtils import linear_regr_eq
classifier.compile(loss=lossF, optimizer='adam', metrics=[linear_regr_eq])

#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#from keras.optimizers import SGD
#opt = SGD(lr=0.01, momentum=0.9)
#classifier.compile(optimizer = opt, loss = 'mean_squared_error', metrics=[linear_regression_equality])

# fit model
history = classifier.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochsNo, verbose=1)

# new instances where we do not know the answer
# https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
# evaluate the model
train_metric = classifier.evaluate(X_train, Y_train, verbose=1)
test_metric = classifier.evaluate(X_test, Y_test, verbose=1)
test_full = classifier.evaluate(X_scaled, Y_scaled, verbose=1)

# log file
logFile = open(file_path + 'model_v2_' + pollutant + '_' + lossF + '_' + str(round(test_full[1]*100,2)).replace('.','_') + '.log', "w+")

# make a prediction
from GEOUtil_CommonUtils import getAcceptedDiff
logFile.write("========================== X_train ==========================\n")
Y_pred = classifier.predict(X_train)
for i in range(len(X_train)):
    print("i=%s,Y=%.6f, Predicted=%.6f, Diff=%.6f, %s" % (str(i).rjust(4,' '), Y_train[i], Y_pred[i,0], abs(Y_train[i]-Y_pred[i,0]),str('').rjust(int(round(abs(((Y_train[i]-Y_pred[i,0]))*10))),'*')))
    logFile.write("i=%s,Y=%.6f, Predicted=%.6f, Diff=%.6f, %s\n" % (str(i).rjust(4,' '), Y_train[i], Y_pred[i,0], abs(Y_train[i]-Y_pred[i,0]),str('').rjust(int(round(abs(((Y_train[i]-Y_pred[i,0]))*10))),'*')))
print('Mean diff =%.6f   Standard deviation =%.6f   max_diff =%.6f   treshhold =%.6f%% ' % (np.mean(Y_train-Y_pred[:,0]), np.std(Y_train-Y_pred[:,0]), (Y_train-Y_pred[:,0]).max(),
                                                                          (sum(abs(Y_train-Y_pred[:,0]<getAcceptedDiff()))/len(Y_train))*100))
logFile.write('Mean diff =%.6f   Standard deviation =%.6f   max_diff =%.6f   treshhold =%.6f%%\n' % (np.mean(Y_train-Y_pred[:,0]), np.std(Y_train-Y_pred[:,0]), (Y_train-Y_pred[:,0]).max(),
                                                                          (sum(abs(Y_train-Y_pred[:,0]<getAcceptedDiff()))/len(Y_train))*100))
logFile.write("\n========================== X_test  ==========================\n")
Y_pred = classifier.predict(X_test)
for i in range(len(X_test)):
    print("i=%s,Y=%.6f, Predicted=%.6f, Diff=%.6f, %s" % (str(i).rjust(4,' '), Y_test[i], Y_pred[i,0], abs(Y_test[i]-Y_pred[i,0]),str('').rjust(int(round(abs(((Y_test[i]-Y_pred[i,0]))*10))),'*')))
    logFile.write("i=%s,Y=%.6f, Predicted=%.6f, Diff=%.6f, %s\n" % (str(i).rjust(4,' '), Y_test[i], Y_pred[i,0], abs(Y_test[i]-Y_pred[i,0]),str('').rjust(int(round(abs(((Y_test[i]-Y_pred[i,0]))*10))),'*')))
print('Mean diff =%.6f   Standard deviation =%.6f   max_diff =%.6f   treshhold =%.6f%% ' % (np.mean(Y_test-Y_pred[:,0]), np.std(Y_test-Y_pred[:,0]), (Y_test-Y_pred[:,0]).max(),
                                                                          (sum(abs(Y_test-Y_pred[:,0]<getAcceptedDiff()))/len(Y_test))*100))
logFile.write('Mean diff =%.6f   Standard deviation =%.6f   max_diff =%.6f   treshhold =%.6f%%\n' % (np.mean(Y_test-Y_pred[:,0]), np.std(Y_test-Y_pred[:,0]), (Y_test-Y_pred[:,0]).max(),
                                                                          (sum(abs(Y_test-Y_pred[:,0]<getAcceptedDiff()))/len(Y_test))*100))
logFile.write("\n========================== X_full  ==========================\n")
Y_pred = classifier.predict(X_scaled)
print('Mean diff =%.6f   Standard deviation =%.6f   max_diff =%.6f   treshhold =%.6f%% ' % (np.mean(Y_scaled-Y_pred[:,0]), np.std(Y_scaled-Y_pred[:,0]), (Y_scaled-Y_pred[:,0]).max(),
                                                                          (sum(abs(Y_scaled-Y_pred[:,0]<getAcceptedDiff()))/len(Y_scaled))*100))
logFile.write('Mean diff =%.6f   Standard deviation =%.6f   max_diff =%.6f   treshhold =%.6f%%\n' % (np.mean(Y_scaled-Y_pred[:,0]), np.std(Y_scaled-Y_pred[:,0]), (Y_scaled-Y_pred[:,0]).max(),
                                                                          (sum(abs(Y_scaled-Y_pred[:,0]<getAcceptedDiff()))/len(Y_scaled))*100))
logFile.write("\n=============================================================\n")

print('%s train: %.3f%%, Test: %.3f%%' % ((classifier.metrics_names[0]).rjust(30,' '), train_metric[0]*100, test_metric[0]*100))
logFile.write('%s train: %.3f%%, Test: %.3f%%\n' % ((classifier.metrics_names[0]).rjust(30,' '), train_metric[0]*100, test_metric[0]*100))
print('%s train: %.3f%%, Test: %.3f%%\n' % ((classifier.metrics_names[1]).rjust(30,' '), train_metric[1]*100, test_metric[1]*100))
logFile.write('%s train: %.3f%%, Test: %.3f%%\n' % ((classifier.metrics_names[1]).rjust(30,' '), train_metric[1]*100, test_metric[1]*100))
print('Full set: %s: %.3f%%   %s: %.3f%%\n' % (classifier.metrics_names[0], test_full[0]*100, classifier.metrics_names[1], test_full[1]*100))
logFile.write('Full set: %s: %.3f%%   %s: %.3f%%\n' % (classifier.metrics_names[0], test_full[0]*100, classifier.metrics_names[1], test_full[1]*100))

# plot loss during training
from matplotlib import pyplot
def plt_loss(history):
    pyplot.title('Loss / Mean Squared Error')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
#plt_loss(history)

#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
classifier.save(file_path + 'model_v2_' + pollutant + '_' + lossF + '_' + str(round(test_full[1]*100,2)).replace('.','_') + '.h5')

def plt_res():
    for band in range(1, 12, 1):
        plt.subplot(3, 4, band)
        plt.plot(X_scaled[:, band -1], Y_pred, 'ro')
        plt.title('Band {}'.format(band))
        plt.ylabel(pollutant)
    plt.show()
#plt_res()

logFile.close()
