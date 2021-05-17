import math
import numpy as np

def bayes_mixture_log_diff(BNmixture, mixture_weight, bn, test_data, k):

    log_diff = 0 

    for pt in test_data:
        true_log = get_point_log(pt, bn)
        learned = 0
        for i in range(k):
            model = BNmixture[i]
            c_prob = 1
            for child, val in enumerate(pt):

                parent = model.parents[child]
                pa_val_lst = [pt[pa] for pa in parent]
                pa_idx = 0 if not pa_val_lst else convert_paIdx(pa_val_lst)
                c_prob *= model.CPTs[child].getTable()[pa_idx][val]

            learned += mixture_weight[i] * c_prob 

        learned_log = math.log(learned)
        log_diff += abs(true_log - learned_log) 

    return log_diff 

def log_pointwise_difference(learned_model, bn, test_data):
    log_diff = 0 

    for pt in test_data:
        learned_log = get_point_log(pt, learned_model)
        true_log = get_point_log(pt, bn)
        log_diff += abs(true_log - learned_log)

    return log_diff

def get_point_log(pt, model):    
    pt_ll = 0

    for child, val in enumerate(pt):
        parent = model.parents[child]
        pa_val_lst = [pt[pa] for pa in parent]
        pa_idx = 0 if not pa_val_lst else convert_paIdx(pa_val_lst)
        pt_ll += math.log(model.CPTs[child].getTable()[pa_idx][val]) 
        
    return pt_ll

def readData(dataFile, is_fully_observed = True):
        
        '''
        Read training/test data
        '''

        df = [] 

        if is_fully_observed:
            
            with open(dataFile) as fname:
                numVar, dataSize = list(map(int, fname.readline().split()))
                
                # col = [i for i in range(numVar)]
                for i in range(dataSize):
                    dataPoint = list(map(int, fname.readline().split()))
                    df.append(dataPoint)
            
        
        else:
            with open(dataFile) as fname:
                numVar, dataSize = list(map(int, fname.readline().split()))
                
                # col = [i for i in range(numVar)]
                for i in range(dataSize):
                    dataPoint = fname.readline().split()
                    df.append(populate_missing_pt(dataPoint))

        return df

def convert_paIdx(pa_val_lst):
    idx = 0
    for i, pa_val in enumerate(reversed(pa_val_lst)):
        idx += pa_val * 2**i
    
    return idx


def populate_missing_pt(pt):

    lst = []

    def helper(s, idx):
        if idx == len(s):
            lst.append(list(map(int, s)))
            return

        if s[idx] == '?':
            for i in ['0', '1']:
                s[idx] = i
                helper(s, idx + 1)
            s[idx] = '?'
        else:
            helper(s, idx + 1)
    
    helper(pt, 0)

    return lst


def countQs(dataFile):
    cnt = 0
    with open(dataFile) as fname:
        numVar, dataSize = list(map(int, fname.readline().split()))
        for i in range(dataSize):
            dataPoint = fname.readline().split()
            c = dataPoint.count('?')
            cnt += 2**c 
    
    return cnt 

def getSE(lst):
    n = len(lst)
    mean = getMean(lst) 
    return math.sqrt(sum( [(x-mean)**2/n for x in lst]))

def getMean(lst):
    return sum(lst)/len(lst)