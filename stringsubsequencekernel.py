from shogun.Features import *
from shogun.Kernel import *
from shogun.Classifier import *
from shogun.Evaluation import *
from modshogun import StringCharFeatures, RAWBYTE
from shogun.Kernel import SSKStringKernel


strings = ['cat', 'doom', 'car', 'boom']
test = ['bat', 'soon']

train_labels  = numpy.array([1, -1, 1, -1])
test_labels = numpy.array([1, -1])

features = StringCharFeatures(strings, RAWBYTE)
test_features = StringCharFeatures(test, RAWBYTE)

# 1 is n and 0.5 is lambda as described in Lodhi 2002
sk = SSKStringKernel(features, features, 1, 0.5)

# Train the Support Vector Machine
labels = BinaryLabels(train_labels)
C = 1.0
svm = LibSVM(C, sk, labels)
svm.train()

# Prediction
predicted_labels = svm.apply(test_features).get_labels()
print predicted_labels