import math
import random

from scipy.optimize import minimize

import numpy as np

from core.BO.gaussian_process import GaussianProcess
from core.BO.acquisition_functions import AcquisitionFunction

import multiprocessing

from core.DIRECT import DIRECTAlgo

num_cores = multiprocessing.cpu_count()

# seed 1 got stuck in local optimum branin
np.random.seed(2609)
random.seed(2609)

num_of_trainingPoints = 40

X = []
Xobs = []
Yobs = []

class BayesianOptimizer():
    def __init__(self, inFunc, initGuess=10, lamda=0.1, kerneltype="ise", numIter=60, isTrans=False, localOptAlgo="L-BFGS-B"):
        self.logs = ""
        self.logDup = ""
        self.logNext = ""
        self.csvLog = ""
        self.logEps = ""
        self.nInitGuess = initGuess
        self.Xobs = []
        self.Yobs = []
        self.function = inFunc
        self.spaces = inFunc.spaces
        self.isTransformation = isTrans
        self.numMultiStart = 5

        self.localOptimAlgo = localOptAlgo

        self.ite = 0
        self.maxIter = numIter

        self.nextArray = []
        self.dup = 0
        self.suggestDup = 0
        self.regretArray = []

        self.dropbest = 0
        self.kernelType = kerneltype
        # Choose kernel
        if self.kernelType == "matern":
            self.GP = GaussianProcess('matern',self.function.input_dim, inFunc, isTrans)  # Choose kernel
        else:
            self.GP = GaussianProcess('ise',self.function.input_dim, inFunc, isTrans)  # Choose kernel

        self.scaleBeta = 1.0
        self.inLamda = lamda
        self.GP.setDiscreteIdx(self.function.discreteIdx)
        self.GP.setCategoricalIdx(self.function.categoricalIdx)

        # Define acquisition function
        acq_func = {}
        acq_func['discrete_idx'] = self.function.discreteIdx
        acq_func['cat_idx'] = self.function.categoricalIdx
        acq_func['dim'] = self.function.input_dim
        acq_func['name'] = "ei" #Default acquisition function

        self.acquisition = AcquisitionFunction(acq_func, inFunc.spaces)
        self.acquisition.setEsigma(lamda)

        self.DIRECT = DIRECTAlgo()
        self.directIte = 200
        self.directEval = 1400
        self.directDeep = 2800

        self.acquisition.setDim(self.function.input_dim)

    def setLamda(self,val):
        self.inLamda = val

    def resetBO(self):
        self.Xobs = []
        self.Yobs = []

        self.ite = 0

        self.nextArray = []
        self.dup = 0
        self.suggestDup = 0
        self.regretArray = []

        self.dropbest = 0

        #self.GP = GaussianProcess('rbs')  # Choose kernel
        if self.kernelType == "matern":
            self.GP = GaussianProcess('matern',self.function.input_dim, self.function, self.isTransformation)  # Choose kernel
        else:
            self.GP = GaussianProcess('ise',self.function.input_dim, self.function, self.isTransformation)  # Choose kernel
        self.GP.setDiscreteIdx(self.function.discreteIdx)
        self.GP.setCategoricalIdx(self.function.categoricalIdx)
        # Define acquisition function
        acq_func = {}
        acq_func['discrete_idx'] = self.function.discreteIdx
        acq_func['cat_idx'] = self.function.categoricalIdx
        acq_func['name'] = 'ucb'
        acq_func['kappa'] = 4.0

        # acq_func['name'] = 'pi'
        acq_func['name'] = 'ei_d'
        # acq_func['name'] = 'tpe'
        # acq_func['name'] = 'pso'
        acq_func['epsilon'] = 0.01
        acq_func['dim'] = self.function.input_dim
        acq_func['inLamda'] = self.inLamda
        self.acquisition = AcquisitionFunction(acq_func, self.spaces)

    def localOptimize(self, input):
        mini = minimize(lambda x: -self.acquisition.acq_kind(x, input[0], input[1], obs=input[4]), input[2], bounds=input[3], method="L-BFGS-B")
        tmpNextX = (mini.x[:]).tolist()
        tmpY = self.acquisition.acq_kind(np.array(tmpNextX), input[0], input[1], obs=input[4])
        return tmpNextX, tmpY

    def _init_guess(self, numOfGuess):
        self.Xobs = []
        for i in range(numOfGuess):
            self.Xobs.append(self.function.randInBounds())
        self.Yobs = [self.function.func(np.array(i)) for i in self.Xobs]
        self.nXobs = self.Xobs.copy()
        for i in range(0,len(self.nXobs)):
            self.nXobs[i] = self.function.normalize(self.nXobs[i])

    def run(self, **args):
        if args.get("method") == "DiscreteBO":
            self._init_guess(self.nInitGuess)
            return self._runBO()

    def _runBO(self):
        self.useOwnGradientBasedOptimizer = False
        self.acquisition.acq_name = 'ucb_opteta'

        self.ite = 0
        lb = []
        ub = []
        for i in self.function.bounds:
            lb.append(i[0])
            ub.append(i[1])
        self.logDebug = ""

        cons = []
        for factor in range(len(self.function.bounds)):
            lower = self.function.bounds[factor][0]
            upper = self.function.bounds[factor][1]
            l = {'type': 'ineq',
                 'fun': lambda x, lb=lower, i=factor: x[i] - lb}
            u = {'type': 'ineq',
                 'fun': lambda x, ub=upper, i=factor: ub - x[i]}
            cons.append(l)
            cons.append(u)
        directBounds = np.array(self.function.bounds)
        self.OptimumIte = -1
        current_Optimum = np.inf

        deltaX = 0.0
        if self.function.input_dim > 1:
            for tmpDim in range(0, self.function.input_dim):
                deltaX += 0.5 * 0.5
            deltaX = np.sqrt(deltaX)
        else:
            deltaX = 0.5

        deltaX = deltaX / self.inLamda

        delt = 0.1
        a = 1.0
        b = 1.0
        dim = self.function.input_dim
        r = 1.0
        for ibound in self.function.bounds:
            r = max(r, ibound[1] - ibound[0])

        max_distance = 0
        for ibound in self.function.bounds:
            max_distance += (ibound[1] - ibound[0]) * (ibound[1] - ibound[0])
        max_distance = np.sqrt(max_distance)
        max_deltaBeta = 1000000

        while (True):
            #####################Theorem1 Srinivas#####################
            precalBetaT = 2.0 * np.log((self.ite + 1) * (self.ite + 1) * math.pi ** 2 / (3 * delt)) + 2 * dim * np.log(
                (self.ite + 1) * (self.ite + 1) * dim * b * r * np.sqrt(np.log(4 * dim * a / delt)))
            BetaT = np.sqrt(precalBetaT) / self.scaleBeta
            ###########################################################
            self.acquisition.setEpsilon(BetaT)

            leftBetaBound = BetaT
            rightBetaBound = BetaT + deltaX

            # Stopping condition
            if (self.ite == self.maxIter + 1):
                break

            # Fit observed X and Y into GP
            self.GP.fit(self.Xobs, self.Yobs)

            # Find the current best point
            current_max_y = np.max(self.Yobs)
            if current_max_y != current_Optimum:
                current_Optimum = current_max_y
                self.OptimumIte = self.ite
            x0 = self.Xobs[np.argmax(self.Yobs)]  # Current best optimal point

            # For n-dim function
            bnds = ()  # Define boundary tuple
            for i in self.function.bounds:
                bnds = bnds + (i,)

            def suggest(eps, initX):
                self.acquisition.setBetaT(BetaT)
                self.acquisition.setEpsilon(eps)
                self.acquisition.setDim(self.function.input_dim)
                self.acquisition.setIte(self.ite + 1)
                ############### L-BFGS-B
                if self.localOptimAlgo == "L-BFGS-B":
                    mini = minimize(lambda x: -self.acquisition.acq_kind(x, self.GP, current_max_y, obs=self.Xobs),
                                    initX, bounds=bnds, method="L-BFGS-B")

                    suggestedX = (mini.x[:]).tolist()
                    # tmpAcq = self.acquisition.acq_kind(suggestedX, self.GP, current_max_y, obs=self.Xobs)
                    tmpAcq = mini.fun * -1.0
                    # ##########################
                    randomPoints = []
                    while (len(randomPoints) < self.numMultiStart):
                        # x0 = self.function.randInBounds()
                        initX = self.function.randUniformInBounds()
                        while initX in randomPoints:
                            # x0 = self.function.randInBounds()
                            initX = self.function.randUniformInBounds()
                        randomPoints.append(initX)

                    tmp = []
                    for i in randomPoints:
                        optX, optA = self.localOptimize([self.GP, current_max_y, i, self.function.bounds, self.Xobs])
                        tmp.append([optX, optA])

                    tmpX = [tmpT[0] for tmpT in tmp]
                    tmpA = [tmpT[1] for tmpT in tmp]
                    tmpX.append(suggestedX)
                    tmpA.append(tmpAcq)

                    suggestedX = tmpX[np.argmax(tmpA)].copy()
                    output_xobs = []

                if self.localOptimAlgo == "DIRECT":
                    # ########################## DIRECT ##########################
                    nextX, _, output_xobs, _ = self.DIRECT.minimize(
                        lambda x: -self.acquisition.acq_kind(x, self.GP, current_max_y, obs=self.Xobs), directBounds,
                        max_iters=self.directIte,
                        max_evals=self.directEval,
                        max_deep=self.directDeep)
                    suggestedX = (nextX[:]).tolist()
                    # mini = minimize(lambda x: -self.acquisition.acq_kind(x, self.GP, current_max_y, obs=self.Xobs),
                    #                 suggestedX, constraints=cons, bounds=directBounds, method="COBYLA")
                    mini = minimize(lambda x: -self.acquisition.acq_kind(x, self.GP, current_max_y, obs=self.Xobs),
                                    suggestedX, bounds=bnds, method="L-BFGS-B")
                    suggestedX = (mini.x[:]).tolist()

                # print("nextX:",nextX)
                return suggestedX, output_xobs

            def roundX(inputX):
                res = inputX.copy()
                for k in self.function.discreteIdx:
                    if self.spaces[k].isUniformlySpaced == False:
                        if abs(inputX[k] - self.spaces[k].getCeil(inputX[k])) > abs(
                                inputX[k] - self.spaces[k].getFloor(inputX[k])):
                            res[k] = self.spaces[k].getFloor(inputX[k])
                        else:
                            res[k] = self.spaces[k].getCeil(inputX[k])
                    else:
                        res[k] = int(round(inputX[k]))
                for k in self.function.categoricalIdx:
                    if self.spaces[k].isUniformlySpaced == False:
                        if abs(inputX[k] - self.spaces[k].getCeil(inputX[k])) > abs(
                                inputX[k] - self.spaces[k].getFloor(inputX[k])):
                            res[k] = self.spaces[k].getFloor(inputX[k])
                        else:
                            res[k] = self.spaces[k].getCeil(inputX[k])
                    else:
                        res[k] = int(round(inputX[k]))
                return res

            def distance(inputX, discreteX):
                return np.linalg.norm(np.array(inputX) - np.array(discreteX))

            # Optimize eps to suggest on discrete
            nextX, mini_xobs = suggest(BetaT, x0)
            if nextX in self.Xobs:
                self.suggestDup += 1

            rX = roundX(nextX)
            BetaNext = BetaT.copy()
            tmpX = rX
            rep = 0
            lenScale = self.GP.getLengthScale()

            ########### CURRENT OPTIMIZE ###########
            curLS = self.GP.libKern.lengthscale.values.tolist()
            curLS = curLS[0]

            # BO
            max_deltaBeta = BetaT * 12
            if rX in self.Xobs and np.max(self.Yobs) < self.function.fmin:  # for Synthetic experiments
                if True:
                    # Define objective
                    foundX = []
                    foundXd = []
                    pairBLS = []

                    def objectiveUnstuck(inputXobj, isBO=True):
                        # print(inputXobj)
                        if isBO:
                            inputXobj = inputXobj[0]
                            deltaBeta = inputXobj[0]
                            newLS = inputXobj[1]
                        else:
                            deltaBeta, newLS = inputXobj

                        pen = 0
                        beta = BetaT + deltaBeta
                        self.GP.setLengthScale(newLS)
                        newX, _ = suggest(beta, x0)
                        roptX = roundX(newX)
                        if roptX in self.Xobs:
                            pen = 1
                        d = np.linalg.norm(np.array(roptX) - np.array(nextX), 2)

                        pairBLS.append([deltaBeta, newLS, d])

                        deltaBeta = deltaBeta / max_deltaBeta
                        d = d / max_distance

                        foundX.append(roptX)
                        foundXd.append(d + deltaBeta + pen)

                        return d + deltaBeta + pen


                    # L-BFGS-B
                    listInitDelta = [[BetaT, curLS]]
                    bndsEps = ((leftBetaBound, max_deltaBeta), (0.07, np.sqrt(r)),)

                    for candidateDelta in listInitDelta:
                        mini = minimize(lambda x: objectiveUnstuck(x, False),
                                        candidateDelta, bounds=bndsEps, method="L-BFGS-B", tol=0.001)
                    rX = foundX[np.argmin(foundXd)]
                    # print("L-BFGS-B Optimized Beta and lengthscale:", pairBLS[np.argmin(foundXd)])
                    # print("L-BFGS-B next optimzed X: ", rX)

            nextY = self.function.func(np.array(rX))

            # Calculate regret for future usages
            print(self.ite, " Minimum: ", np.max(self.Yobs) * self.function.ismax * -1.0, " at x:",
                  self.Xobs[np.argmax(self.Yobs)], " nextX:", rX, " nextY:", nextY * self.function.ismax,
                  " Xobs size:", len(self.Xobs))

            # Update Xobs Yobs
            if rX not in self.Xobs:
                self.Xobs.append(rX)
                self.Yobs.append(nextY)
                self.acquisition.isSame = False;
                # dup=0
            else:
                self.acquisition.isSame = True;
                self.dup += 1

            # Increase counter
            self.ite += 1
        return self.Xobs[np.argmax(self.Yobs)], self.Xobs, self.Yobs, self.ite
