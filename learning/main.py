from bayesian_model import BN, CPT
from helper import log_pointwise_difference, readData, bayes_mixture_log_diff, getMean, getSE
from parameter_learning import FOD_learner, POD_EM_learner, mixture_random_Bayes
import sys
import time
import random
import numpy as np

uai_file = sys.argv[1]
task_id = int(sys.argv[2])
train_file = sys.argv[3]
test_file = sys.argv[4]

dataset = 'dataset' + uai_file.split('.')[0]
modelPath = 'hw5-data/' + dataset + '/' + uai_file
trainPath = 'hw5-data/' + dataset + '/' + train_file
testPath = 'hw5-data/' + dataset + '/' + test_file

bn = BN() 
bn.readModel(modelPath)
if task_id in [1,3]:
    train_df = readData(trainPath, is_fully_observed = True)
else:
    train_df = readData(trainPath, is_fully_observed = False)

test_df = readData(testPath, is_fully_observed = True)

print('-' * 50)

if task_id == 1:
    model = FOD_learner()
    model.estimate(train_df, bn)
    print('log likelihood difference: ', log_pointwise_difference(model, bn, test_df))
elif task_id == 2:
    lst = []
    for i in range(5):
        np.random.seed(i)
        model = POD_EM_learner(seed_number = i)
        model.estimate(train_df, bn)
        lst.append(log_pointwise_difference(model, bn, test_df))
    mean = getMean(lst)
    se = getSE(lst)
    print('mean log likelihood difference =  {} with standard deviation = {} '.format(mean, se))
else: 
    lst = []
    k = int(input('Input number of mixture Bayesian networks: '))
    for i in range(5):
        np.random.seed(i)
        model = mixture_random_Bayes(k, seed_number = i)
        model.estimate(train_df, bn)
        lst.append(bayes_mixture_log_diff(model.mixtureBN, model.weight_components, bn, test_df, k))
    mean = getMean(lst)
    se = getSE(lst)
    print('mean log likelihood difference =  {} with standard deviation = {} '.format(mean, se))

print('-' * 50)


