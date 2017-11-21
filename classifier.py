]
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:22:35 2017

@author: insane
"""
import sklearn as sk
from sklearn import svm
import csv
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sknn.mlp import Classifier, Layer
from sklearn.model_selection import GridSearchCV
from theano import function, config, shared, tensor
from keras.models import model_from_json
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import MaxPooling1D 
from keras.utils import np_utils
from keras.layers import Dropout
from keras.optimizers import SGD
from sklearn import svm
#from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras import models
from keras import optimizers
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn import ensemble 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression,SGDClassifier
from sklearn.ensemble import VotingClassifier ,RandomForestClassifier,GradientBoostingClassifier ,ExtraTreesClassifier, AdaBoostClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report,confusion_matrix

'''
This code makes use of training and testing feature matrix obtianed to train and test an ExtraTreesClassifier, a RandomForestClassifier and also a MLPClassifier.
'''

#Load train and test data
exampleFile = open('/home/insane/Documents/data/csv/rgb/train.csv')      #read csv  files, train data, train label, test data and test label
exampleReader = csv.reader(exampleFile)                                  # train data and test data are feature matrices obtained from the other code
exampleData = list(exampleReader)
X= np.array(exampleData)
X = np.float32(X)

exampleFile1 = open('/home/insane/Documents/data/csv/rgb/train_label.csv')
exampleReader1 = csv.reader(exampleFile1)
exampleData1 = list(exampleReader1)
Y = np.array(exampleData1)
Y = np.int16(Y)
y = np.ravel(Y)
y2 = np_utils.to_categorical(y, 8)

exampleFile2 = open('/home/insane/Documents/data/csv/rgb/test.csv')
exampleReader2 = csv.reader(exampleFile2)
exampleData2 = list(exampleReader2)
X1= np.array(exampleData2)
X1= np.float32(X1)

exampleFile3 = open('/home/insane/Documents/data/csv/rgb/test_label.csv')
exampleReader3 = csv.reader(exampleFile3)
exampleData3 = list(exampleReader3)
Y1 = np.asarray(exampleData3)
Y1 = np.int16(Y1)
y1 = np.ravel(Y1)
y3 = np_utils.to_categorical(y1, 8)


###Training and testing a MLP classifier. using keras

#ns = (800,205,1)
#ns1 = (1888,205,1)
#
#X3=np.reshape(X,ns)
#X4= np.reshape(X1,ns1)
#rms = optimizers.rmsprop(lr = 0.001,rho = 0.5, epsilon = 1e-7,decay= 0)
#model = Sequential()
#model.add(Dense(16, input_dim=205, init='uniform', activation='relu'))
#model.add(Dense(128, init='uniform', activation='relu'))
#model.add(Dense(8, init='uniform', activation='sigmoid'))
#model.add(Conv1D(64,3,input_shape=(205,1),border_mode = 'same', activation='relu'))
#model.add(MaxPooling1D(pool_length =2))
#model.add(Conv1D(128,3,border_mode = 'same', activation='relu'))
#model.add(MaxPooling1D(pool_length =2))
#model.add(Flatten())
#model.add(Dense(128,activation ='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(8,activation = 'softmax'))
## Compile model
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
### Fit the model
#
#model.fit(X4, y3, validation_data=(X3,y2), nb_epoch=45, batch_size = 5)
#### calculate predictions
#model_json = model.to_json()
#with open("model1.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")
# 
#
#model.save('classification_model.h5')
## round predictions



#acc = []
##i=1
#for i in range(149,160):
# clf3 = ensemble.ExtraTreesClassifier(n_estimators=2100,max_features=i, max_depth=None,  min_samples_split=2, random_state=0)
# clf3.fit(X1, y1)  
# pred3 = clf3.predict(X)
# acc.append(accuracy_score(y,pred3))
#clf4 = ensemble.BaggingClassifier(bootstrap=True,n_jobs = -1)
#clf4.fit(X1, y1)  
#pred4 = clf4.predict(X)
#acc4 = accuracy_score(y,pred4)
 


#Training and testing invidual as well as ensembles of classifiers.
clf3 = RandomForestClassifier(n_estimators=2000, n_jobs=-1)               #Initializing the classifiers
#clf4 = GradientBoostingClassifier(n_estimators=15000,learning_rate=0.0001,max_features=30)
clf5 = ExtraTreesClassifier(n_estimators=2000, max_features=100,n_jobs=-1)
#
#est = [('rf',clf3),('gb',clf4)]
#est1 = [('rf',clf3),('et',clf5)]
#est2 = [('gb',clf4),('et',clf5)]

#eclf1 = VotingClassifier(est,voting='hard',weights = [2,1])
#eclf1.fit(X1,y1)
#pred1 = eclf1.predict(X)
#acc = accuracy_score(y,pred1)
#
#eclf2 = VotingClassifier(est1,voting='hard',weights = [2,1])
#eclf2.fit(X1,y1)
#pred2 = eclf2.predict(X)
#acc2 = accuracy_score(y,pred2)
#                                                                         #Training an ensemble of classifiers
#eclf3 = VotingClassifier(est2,voting='hard',weights = [2,1])
#eclf3.fit(X1,y1)
#pred3 = eclf3.predict(X)
#acc3 = accuracy_score(y,pred3)
#
#eclf4 = VotingClassifier(est,voting='hard',weights = [1,2])
#eclf4.fit(X1,y1)
#pred4 = eclf4.predict(X)
#acc4 = accuracy_score(y,pred4)
#
#eclf5 = VotingClassifier(est1,voting='hard',weights = [1,2])
#eclf5.fit(X1,y1)
#pred5 = eclf5.predict(X)
#acc5 = accuracy_score(y,pred5)

#eclf6 = VotingClassifier(est2,voting='hard',weights = [1,2])
#eclf6.fit(X1,y1)
#pred6 = eclf6.predict(X)
#acc6 = accuracy_score(y,pred6)

#print confusion_matrix(y,pred6)
clf3.fit(X1, y1)
#clf4.fit(X1, y1)                                                         #Training the individual classifiers
clf5.fit(X1, y1)

pred3 = clf3.predict(X)
#pred4 = clf4.predict(X)                                                  #Prediction using the trainined classifiers
pred5 = clf5.predict(X)

acc7 = accuracy_score(y,pred3)
#acc8 = accuracy_score(y,pred4)                                           #Calculating accuracy of the models 
acc9 = accuracy_score(y,pred5)

print confusion_matrix(y,pred3)
print confusion_matrix(y,pred5)                                          


