from Factor import Factor
from GraphicalModel import GraphicalModel
import sys 
import math
def variableElimination():
    '''
    Main function to run variable elimination algorithm 
    '''
    inputFile = sys.argv[1]
    evidFile = sys.argv[2]

    gm = GraphicalModel()
    gm.read(inputFile)

    gm.instantiate(evidFile)
    gm.order()
    for i in range(len(gm.minDegreeOrder)):
        var = gm.minDegreeOrder[i]
        cluster = []

        for f in gm.factors:
            if f.contains(var):
                cluster.append(f)

        cluster.sort(key = lambda x: x.getTableSize())
        clusterFactor = cluster[0]
       
        for i in range(1, len(cluster)):
            clusterFactor = gm.product(clusterFactor, cluster[i])

        for c in cluster:
            gm.factors.remove(c)

        marginalFactor = gm.sumout(clusterFactor, var)
        gm.factors.append(marginalFactor)

    prob = 1
    for f in gm.factors:
        prob *= f.getTable()[0]

    return math.log(prob, 10)

print('Partition function in log10: ', variableElimination())