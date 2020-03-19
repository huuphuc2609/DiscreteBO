import time
import numpy as np

from core.BayesianOptimizer import BayesianOptimizer
from core.Space import Space

class discreteBranin:
    def __init__(self):
        self.numCalls = 0
        self.input_dim = 2
        self.bounds = [(-5, 10),(0, 15)]
        self.fmin = 0.497910
        self.min = [-3,12]
        self.ismax = -1
        self.name = 'DiscreteBranin'
        self.discreteIdx = [0,1]
        self.categoricalIdx = [0, 1]
        self.spaces = [Space(-5, 10, True), Space(0, 15, True)]


    def _interfunc(self,X):
        X=np.asarray(X)

        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
        a=1.0
        b=5.1/(4.0*np.pi**2)
        c=5.0/np.pi
        r=6.0
        s=10.0
        t=1.0/(8.0*np.pi)
        fx=a*(x2-b*x1*x1+c*x1-r)**2+s*(1-t)*np.cos(x1)+s
        return fx * self.ismax

    def func(self,X):
        self.numCalls += 1
        for xi in X:
            if xi != int(xi):
                raise Exception("Invalid input")
        return self._interfunc(X)

    def randInBounds(self):
        rand = []
        for i in self.bounds:
            tmpRand = int(np.random.uniform(i[0], i[1]+1))
            rand.append(tmpRand)
        return rand

    def randUniformInBounds(self):
        rand = []
        for i in self.bounds:
            tmpRand = np.random.uniform(i[0], i[1])
            rand.append(tmpRand)
        return rand

    def normalize(self,x):
        val = []
        for i in range(0,self.input_dim):
            val.append((x[i] - self.bounds[i][0])/(self.bounds[i][1] - self.bounds[i][0]))
        return val

    def denormalize(self,x):
        val = []
        for i in range(0, self.input_dim):
            val.append(1.0 * (x[i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]))
        return val

############################## DISCRETE BO ##############################
myfunction = discreteBranin()
for i in range(0, 10):
    BO = BayesianOptimizer(myfunction)
    BOstart_time = time.time()
    bestBO, box, boy, ite = BO.run(method="DiscreteBO")
    BOstop_time = time.time()
    ite = ite-1
    print("Iter:", i, " Discrete BO x: ", bestBO, " y:", -1.0*myfunction.func(bestBO), " ite:", ite, " time: --- %s seconds ---" % (BOstop_time - BOstart_time))
    BO.resetBO()



