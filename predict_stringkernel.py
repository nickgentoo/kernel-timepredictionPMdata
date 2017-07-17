from sklearn.datasets import load_digits
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from load_dataset_4stringkernels import load_dataset
from string_kernel import StringKernel
from sklearn.model_selection import GridSearchCV

import sys

if __name__ == '__main__':
    cur_f = __file__.split('/')[-1]
    if len(sys.argv) < 4:
        print >> sys.stderr, 'usage: ' + cur_f + ' <maximal subsequence length> <lambda (decay)> <dataset> '
        sys.exit(1)
    else:
        subseq_length = int(sys.argv[1])
        lambda_decay = float(sys.argv[2])
        dataset=sys.argv[3]

kernel=StringKernel(subseq_length,lambda_decay)

(X_train, y_train), (X_test,y_test)= load_dataset(dataset)

#svr = GridSearchCV(SVR(kernel='precomputed'), cv=5, param_grid={"C": [1e-4,1e-3,1e-2,1e-1,1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]}, scoring="neg_mean_squared_error")

#X_train= X_train[:10000]
#y_train= y_train[:10000]

average=np.average(y_train)
print "dividing target by", average
y_train=y_train/average
y_test=y_test/average

kernel_train = kernel.string_kernel(X_train, X_train)
kernel_test = kernel.string_kernel(X_test, X_train)
print "Kernel computed"

for C in  np.logspace(-12,3,num=16,base=10,dtype='float'):
  for eps in [0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    svr=SVR(kernel='precomputed',C=C,epsilon=eps)
    svr.fit(kernel_train, y_train)

    #
    # print("Best parameters set found on development set:")
    # print(svr.best_params_)
    # print("Grid scores on development set:")
    # print("")
    # means = svr.cv_results_['mean_test_score']
    # stds = svr.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, svr.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # #svr = SVR(kernel='precomputed')


    print "Kernel computed and model trained"
    #kernel_test = np.dot(X_test, X_train[svc.support_, :].T)
    y_pred = svr.predict(kernel_test)
    print "C: ", C, "epsilon:", eps
    #print (mean_squared_error(y_test, y_pred)/((24.0*3600)**2) )**0.5
    print 'Rooted mean squared error: %0.3f MAE: %0.3f' % ( (mean_squared_error(y_test*average, y_pred*average)/((24.0*3600)**2) )**0.5, mean_absolute_error(y_test*average, y_pred*average)/(24.0*3600) )