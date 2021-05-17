import collections    
import random 
import math 
from GraphicalModel import GraphicalModel
from Factor import Factor 
import copy

def SVEC(G, w, N, Q_type):
    '''
    Sampling-based Variable Elimination and Conditioning
    
    G: evidence instatiated Markov or a Bayesian network
    w: cutset bound
    N: number of samples
    Q: proposal distribution. 'uniform' or 'adaptive'. Default option is uniform
    '''

    Z = 0 
    X_cutset = wCutset(G, w)
    weights = [] 
    X_samples = []
    i = 0

    if Q_type.lower() == 'uniform':
        Q_dist = getUniformQ(G, X_cutset)
        while i < N:
            X_sample = generateSample(X_cutset, Q_dist) 
            tmpG = copy.deepcopy(G)
            tmpG.instantiateSample(X_sample)
            P_val = variableElimination(tmpG)
            if P_val == math.inf: continue 
            Q_val = getQVal(Q_dist, X_sample)
             
            w = P_val - Q_val
            Z += w
            i += 1
    elif Q_type.lower() == 'adaptive': 
        j = 0
        while i < N:
            if i == 0:
                Q_dist = getUniformQ(G, X_cutset)
                while j <  min(N, 100):
                    X_sample = generateSample(X_cutset, Q_dist)  
                    tmpG = copy.deepcopy(G)
                    tmpG.instantiateSample(X_sample)
                    P_val = variableElimination(tmpG)
                    if P_val == math.inf: continue 
                    Q_val = getQVal(Q_dist, X_sample)
                    w = P_val - Q_val
                    X_samples.append(X_sample) 
                    weights.append(w) 
                    Z += w
                    j += 1
                i = j

            else:
                Q_dist = updateQ(Q_dist, weights, X_samples)
                weights = []
                X_samples = []
                while j < min(N, i+100): 
                    X_sample = generateSample(X_cutset, Q_dist)  
                    tmpG = copy.deepcopy(G)
                    tmpG.instantiateSample(X_sample)
                    P_val = variableElimination(tmpG)
                    if P_val == math.inf: continue
                    Q_val = getQVal(Q_dist, X_sample)
                    w = P_val - Q_val
                    X_samples.append(X_sample) 
                    weights.append(w)
                    Z += w
                    j += 1
                i = j
    return Z/N  

def getUniformQ(G, X):
    '''
    Return a uniform proposal distribution Q over set of variables X
    '''
    dist = collections.defaultdict(dict)
    for x in X:
        for i in range(G.factors[x].getTableSize()):
            dist[x][i] = 1/G.factors[x].getTableSize()
    return dist 

def getQVal(Q_dist, X_sample):
    '''
    Get P(Q = X) with Q is proposal distribution over X and X is sample set
    '''
    val = 0
    for x in X_sample:
        val += math.log(Q_dist[x][X_sample[x]], 10) 
    
    return val

def updateQ(Q_dist, weights, X_samples):
    '''
    Q_dist: previous proposal distribution need to be updated
    weights: list of weight of previous samples patch 
    X_samples: list of previous samples patch  
    '''
    numerator = sum([w for w in weights])
    for x in Q_dist:
        for val in Q_dist[x]:
            denominator = 0

            for i, sample in enumerate(X_samples):
                if x in sample and sample[x] == val:
                    denominator += weights[i]

            Q_dist[x][val] = denominator/numerator

    return Q_dist

def generateSample(X, Q_dist):

    '''
    Generate sample with uniform proposal distribution

    X: set of variables to generate sample
    '''
    sample = {}

    for x in X:
        cumulDist = [0] * (len(Q_dist[x])+ 1)
        for i, val in enumerate(sorted(Q_dist[x])):
            cumulDist[i+1] = Q_dist[x][val] + cumulDist[i]
        r = random.random() * cumulDist[-1]
        for i in range(1, len(cumulDist)):
            if cumulDist[i] > r:
                sample[x] = i - 1
                break

    return sample

def wCutset(G, w):
    '''
    Find the w-cutset of a graphical model

    G: EVIDENCE INSTANTIATED Markov/ Bayesian network
    w: cutset integer
    '''

    G.order()
    X_cutset = set()
    width = 0
    C = G.tree_decomposition.copy()

    for cluster in C.values():
        width = max(len(cluster), width)
    
    
    while w + 1 < width:
        max_freq = 0
        width = 0

        d = {}

        for bucket in C.values():
            for var in bucket:
                d[var] = d.get(var, 0) + 1
                if d[var] >= max_freq:
                    max_freq = d[var]
                    removedVar = var

        X_cutset.add(removedVar)

        for var in C:
            if removedVar in C[var]:
                C[var].remove(removedVar)
            width = max(len(C[var]), width)

    return X_cutset


def variableElimination(G):
    '''
    Main function to run variable elimination algorithm 
    '''
    G.order()
    # print(G.getOrder())
    for i in range(len(G.minDegreeOrder)):
        var = G.minDegreeOrder[i]
        cluster = []

        for f in G.factors:
            if f.contains(var):
                cluster.append(f)

        # print([f.getScope() for f in cluster])
    
        cluster.sort(key = lambda x: x.getTableSize())
        clusterFactor = cluster[0]
        
        for i in range(1, len(cluster)):
            clusterFactor = G.product(clusterFactor, cluster[i])

        for c in cluster:
            G.factors.remove(c)

        marginalFactor = G.sumout(clusterFactor, var)
        G.factors.append(marginalFactor)

    prob = 0
    for f in G.factors:
        prob += math.log(f.getTable()[0], 10)
        # print(prob)

    return prob

def log_error(Z, Z_hat):
    return (math.log(Z, 10) - math.log(Z_hat, 10))/ math.log(Z, 10)

def getSE(lst):
    n = len(lst)
    mean = getMean(lst) 
    return math.sqrt(sum( [(x-mean)**2/n for x in lst] ))

def getMean(lst):
    return sum(lst)/len(lst)
    
    