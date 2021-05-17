from bayesian_model import BN, CPT
from helper import convert_paIdx
import collections
import heapq 
import numpy as np
import random 
import math


class FOD_learner:
    def __init__(self):
        self.parents = {}
        self.children = []
        self.CPTs = {}         

    def estimate(self, data , bn):
        self.parents = bn.parents
        self.children = bn.children
        
        for c in self.children:
            self.CPTs[c] = self.node_MLE(c, self.parents[c], data)

    def node_MLE(self, child, parent, data):
        
        tbl = [[10**-5 for _ in range(2)] for _ in range(2**len(parent))]
        d = {}

        for pt in data:
            pa_val_lst = [pt[pa] for pa in parent]
            pIdx = 0 if not pa_val_lst else convert_paIdx(pa_val_lst)
            d[(pIdx, pt[child])] = d.get((pIdx, pt[child]), 0) + 1

        for pIdx, cIdx in d:
            tbl[pIdx][cIdx] = d[(pIdx, cIdx)]

        tmp_cpt = CPT(parent, child, np.array(tbl))
        tmp_cpt.normalize()

        return tmp_cpt


class POD_EM_learner:
    def __init__(self, seed_number = 0):
        
        self.seed_number = seed_number
        self.parents = {}
        self.children = []
        self.CPTs = {}
    
    def estimate(self, data, bn):
        np.random.seed(self.seed_number)
        self.parents = bn.parents
        self.children = bn.children 

        self.initialize(self.seed_number)
        
        # cap iteration to 20
        for i in range(20):
            weights = self.E_step(data)
            self.M_step(weights, data)

    def initialize(self, seed_number):

        np.random.seed(seed_number)

        for c in self.children:
            parent = self.parents[c]
            rowSize = 2**len(parent)
            
            tbl = []
            for r in range(rowSize):
                c0 = random.random()
                tbl.append([c0, 1-c0])

            self.CPTs[c] = CPT(parent, c, np.array(tbl))
        
    def E_step(self, data):

        weight = []
        for i, sample_pts in enumerate(data):
            pt_weights = []
            for pt in sample_pts:
                w = 1
                for c, val in enumerate(pt):
                    pa_val_lst = [pt[pa] for pa in self.parents[c]]
                    pIdx = 0 if not pa_val_lst else convert_paIdx(pa_val_lst)
                    w *= self.CPTs[c].getTable()[pIdx][val]
                pt_weights.append(w)
            
            # normalize
            s = sum(pt_weights)
            for i in range(len(pt_weights)):
                pt_weights[i] /= s
                
            weight.append(pt_weights)

        return weight 

    def M_step(self, weights, data):

        for c in self.children:
            self.CPTs[c] = self.cpt_update(c, self.parents[c], weights, data, self.CPTs[c].getTable())          
        
    def cpt_update(self, child, parent, weights, data, tbl):

        # if not self.CPTs:
        #     tbl = [[10**-5 for _ in range(2)] for _ in range(2**len(parent))]  
                
        Pxu = {}
        Pu = {}
        for ws, pts in zip(weights, data):
            for w, pt in zip(ws, pts):
                pa_val_lst = [pt[pa] for pa in parent]
                pIdx = 0 if not pa_val_lst else convert_paIdx(pa_val_lst)
                Pxu[(pIdx, pt[child])] = Pxu.get((pIdx, pt[child]), 0) + 1
                Pu[pIdx] = Pu.get(pIdx, 0) + 1
        
        for pIdx, cIdx in Pxu:
            tbl[pIdx][cIdx] = Pxu[(pIdx, cIdx)]/Pu[pIdx]
        
        tmp_cpt = CPT(parent, child, np.array(tbl))
        tmp_cpt.normalize()
        
        return tmp_cpt
        
class mixture_random_Bayes:
    def __init__(self, k, seed_number = 0):
        self.seed_number = seed_number
        self.k = k 
        self.mixtureBN = []  
        self.weight_components = np.ones([1, k])

        self.CPTs = {}
        self.children = []
        self.parents = {} 

    def estimate(self, data, bn):
        
        np.random.seed(self.seed_number)
        
        self.children = bn.children
        self.parents = {i: [] for i in self.children}
        self.initialize(bn.numVars, self.k)
        for _ in range(20):
            ws = self.E_step(data)
            self.M_step(ws, data)


    def E_step(self, data):
        
        dataWeights = []
        for pt in data:
            ws = []
            for i in range(self.k):
                w = 1
                for c, val in enumerate(pt):
                    pa_val_lst = [pt[pa] for pa in self.mixtureBN[i].parents[c]]
                    pIdx = 0 if not pa_val_lst else convert_paIdx(pa_val_lst)
                    w *= self.mixtureBN[i].CPTs[c].getTable()[pIdx][val]
                ws.append(w)
            dataWeights.append(ws)
        
        return np.array(dataWeights)

    def M_step(self, ws, data):
        # update weight components
        self.weight_components = np.mean(ws, axis = 0)

        # update CPTs for each bayesian model
        for i in range(self.k):
            self.mixtureBN[i] = self.update_BNmixture(self.mixtureBN[i], data)            

    def initialize(self, noVar, k):

        np.random.seed(self.seed_number)
        # initialize k bayesian network
        for i in range(k):
            bn = BN()
            bn.randomBN(noVar, seed_number = i)
            self.mixtureBN.append(bn)

        # initialize weight component
        self.weight_components = np.random.dirichlet(np.ones(k), size = 1)

    def update_BNmixture(self, bayes_nw, data):
        
        elist = []
        pq = [] 
        dct = {node: i for i, node in enumerate(bayes_nw.children)}
        # construct complete graph with mutual information
        for n1 in bayes_nw.children:
            for n2 in bayes_nw.children:
                if n1 != n2:
                    mi = self.get_mi(n1, n2, data)
                    heapq.heappush(pq, Edge(-mi, n1, n2))
                    heapq.heappush(pq, Edge(-mi, n2, n1))
        
        # Kruskal algorithm to get spanning tree 
        n = bayes_nw.numVars
        ds = DisjSet(bayes_nw.numVars)
        set1 = set2 = 0 

        while len(elist) < n - 1:
            edge = heapq.heappop(pq)
            set1 = ds.find(dct[edge.get_node1()]) 
            set2 = ds.find(dct[edge.get_node2()])

            if set1 != set2:
                elist.append([edge.get_node1(), edge.get_node2()])
                ds.union(set1, set2)        
        
        # Convert to rooted tree and contruct a parent, child lst         
        root = random.randint(0, bayes_nw.numVars)
        q = collections.deque() 
        q.append(root)
        s = set() 
        adjList = collections.defaultdict(list)
        parents = collections.defaultdict(list)
        
        for n1, n2 in elist: 
            adjList[n1].append(n2)
            adjList[n2].append(n1)
        
        cur_pa = []
        while q:
            node = q.popleft() 
            if node not in s: 
                parents[node] = cur_pa 
                s.add(node)
                q.extend(adjList[node])
        
        # Learn Bayesian tree network
        tree_bn = BN()
        tree_bn.parents = parents
        tree_bn.numVars = bayes_nw.numVars 
        tree_bn.cardinalities = bayes_nw.cardinalities
        tree_bn.children = bayes_nw.children 

        mle = FOD_learner()
        mle.estimate(data, tree_bn)
        tree_bn.CPTs = mle.CPTs

        return tree_bn

    def get_mi(self, x, u, data):
        Pxu = collections.defaultdict(int)
        Px = collections.defaultdict(int)
        Pu = collections.defaultdict(int)
        n = len(data)
        mi = 0
        for pt in data:
            x_val = pt[x]
            u_val = pt[u] 
            Pxu[(x_val, u_val)] += 1
            Px[x_val] += 1
            Pu[u_val] += 1
        
        for x, u in Pxu:
            Pxu[(x,u)] /= n 
            Px[x] /= n 
            Pu[u] /= n 
            mi += Pxu[(x,u)] * math.log(Pxu[(x,u)]/ (Px[x] * Pu[u]))

        return mi 

class DisjSet:
    def __init__(self, num_elem):
        self.s = [-1] * num_elem 
    
    def union(self, r1, r2):
        if self.s[r2] <  self.s[r1]:
            self.s[r1] = r2 
        else:
            if self.s[r1] == self.s[r2]:
                self.s[r1] -= 1
            self.s[r2] = r1
    
    def find(self, x):
        if self.s[x] < 0:
            return x
        else:
            return self.find(self.s[x])
    
class Edge:
    
    def __init__(self, w, n1 , n2):
        self.n1 = n1
        self.n2 = n2
        self.w = w 
    
    def get_weight(self):
        return self.w
    
    def get_node1(self):
        return self.n1 

    def get_node2(self):
        return self.n2

    def __lt__(self, other):
        return self.w < other.w