import math
import numpy as np
import scipy
from scipy.stats import norm

class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, acq, inSpaces):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.acq = acq
        self.spaces = inSpaces
        self.acq_name = self.acq['name']
        self.idxDiscrete = self.acq['discrete_idx']
        self.idxCat = self.acq['cat_idx']
        self.stillGood = True
        self.isSame = False
        self.decrement = 2
        self.currentEpsilon = 0.01

        self.dens_good = None
        self.dens_bad = None
        self.varType = ""
        for i in range(0,len(self.idxCat)):
            self.varType += 'u'

    def setEpsilon(self, val):
        self.acq['epsilon'] = val

    def setBetaT(self, val):
        self.acq['betaT'] = val

    def setIte(self, val):
        self.ite = val

    def setDim(self, val):
        self.dim = val

    def setEsigma(self, val):
        self.esigma = val

    @staticmethod
    def resampleFromKDE(kde, size):
        n, d = kde.data.shape
        indices = np.random.randint(0, n, size)
        cov = np.diag(kde.bw) ** 2
        means = kde.data[indices, :]
        norm = np.random.multivariate_normal(np.zeros(d), cov, size)
        return np.transpose(means + norm)

    def getEpsilon(self):
        return self.acq['epsilon']

    def acq_kind(self, x, gp, y_max, **args):
        if np.any(np.isnan(x)):
            return 0
        if self.acq_name == 'ucb':
            return self._ucb(x, gp, self.acq['epsilon'])
        if self.acq_name == 'lcb':
            return self._lcb(x, gp, self.acq['kappa'])
        if self.acq_name == 'ei':
            return self._ei(x, gp, y_max, self.idxDiscrete, self.acq['epsilon'], self.stillGood, self.currentEpsilon)
        if self.acq_name == 'eiT':
            return self._eiT(x, gp, y_max, self.idxDiscrete, self.acq['epsilon'], self.stillGood, self.currentEpsilon)
        if self.acq_name == 'pi':
            return self._pi(x, gp, y_max, self.acq['epsilon'])
        if self.acq_name == 'mu':
            return self._mu(x, gp, self.acq['epsilon'])
        if self.acq_name == 'tpe':
            return self._tpe(x, args.get("goodKDE"), args.get("badKDE"))
        if self.acq_name == 'pso':
            return self._pso(x, gp, y_max, args.get("goodKDE"), args.get("badKDE"), self.acq['epsilon'])
        if self.acq_name == 'ei_d':
            return self._ei_d(x, gp, y_max, self.idxDiscrete, self.spaces, self.acq['epsilon'], self.currentEpsilon, self.stillGood, self.isSame, args.get("obs"))
        if self.acq_name == 'ei_new':
            return self._ei_new(x, gp, y_max, self.idxDiscrete, self.idxCat, self.dens_good, self.dens_bad, self.spaces, self.acq['epsilon'], self.currentEpsilon, self.stillGood, self.isSame, args.get("obs"))
        if self.acq_name == 'eiTest':
            return self._eiTest(x, gp, y_max, self.idxDiscrete, self.spaces, self.acq['epsilon'], self.currentEpsilon, self.stillGood, self.isSame, args.get("obs"))
        if self.acq_name == 'eiRound':
            return self._eiRound(x, gp, y_max, self.idxDiscrete, self.spaces, self.acq['epsilon'], self.currentEpsilon, self.stillGood, self.isSame, args.get("obs"))
        if self.acq_name == 'ucb_d':
            return self._ucb_d(x, gp, self.dim, self.ite, self.acq['epsilon'], self.acq['betaT'], self.idxDiscrete, self.idxCat, self.spaces)
        if self.acq_name == 'ucb_d2':
            return self._ucb_d2(x, gp, self.acq['epsilon'])
        if self.acq_name == 'lcb_d':
            return self._lcb_d(x, gp, y_max, self.idxDiscrete, self.acq['kappa'])
        if self.acq_name == 'ucb_optv':
            return self._ucb_optv(x, gp, self.dim, self.ite, self.acq['epsilon'])
        if self.acq_name == 'ucb_opteta':
            return self._ucb_opteta(x, gp, self.dim, self.ite, self.acq['epsilon'], self.acq['betaT'], self.idxDiscrete, self.idxCat, self.spaces, args.get("obs"), self.esigma)

    @staticmethod
    def _pi(x, gp, fMax, epsilon):
        mean, _, var = gp.predictScalar(x)
        #var[var < 1e-10] = 0
        std = np.sqrt(var)
        Z = (mean - fMax - epsilon) / std
        result = np.matrix(scipy.stats.norm.cdf(Z))
        return result

    @staticmethod
    def _ei(x, gp, fMax, discrete_idx, epsilon, stillGood, currentEps):
        # if stillGood:
        #     currentEps = currentEps
        # else:
        #     currentEps = currentEps/2
        # actualEpsilon = currentEps
        # epsilon = actualEpsilon
        epsilon = 0.01
        # for i in discrete_idx:
        #     x[i] = int(round(x[i]))
        #mean, _, var = gp.predictScalar(x)
        mean, _, var = gp.predictScalarLib(x)
        #mean, _, var = gp.predictScalarTrans(x)
        #print("mean: ", mean)
        #print("var: ", var)
        #mean, _, var = gp.predictCholeskyScalar(x)
        var2 = np.maximum(var, 1e-4 + 0 * var)
        #var[var < 1e-10] = 0
        std = np.sqrt(var2)
        Z = (mean - fMax - epsilon) / (std)
        # result = np.matrix((mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + np.multiply(np.matrix(std.ravel()), (
        #     np.matrix(scipy.stats.norm.pdf(Z)))).ravel())
        result = (mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        return result

    @staticmethod
    def _eiT(x, gp, fMax, discrete_idx, epsilon, stillGood, currentEps):
        epsilon = 0.01
        mean, _, var = gp.predictScalarTrans(x)
        # mean, _, var = gp.predictCholeskyScalar(x)
        var2 = np.maximum(var, 1e-16 + 0 * var)
        # var[var < 1e-10] = 0
        std = np.sqrt(var2)
        Z = (mean - fMax - epsilon) / (std)
        result = (mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        return result

    @staticmethod
    def _ucb(x, gp, beta):
        # if gp.checkExistInObs(x):
        #     return 0
        #mean, _, var = gp.predictScalar(x)
        mean, _, var = gp.predictScalarLib(x)
        # mean, _, var = predictCholeskyScalar(x, Xm, Ym)
        #var[var < 1e-10] = 0
        std = np.sqrt(var)
        result = np.matrix(mean + beta * std)
        return result

    @staticmethod
    def _ucb_d(x, gp, dim, ite, inEps, betaT, discrete_idx, cat_idx, spaces):
        # if gp.checkExistInObs(x):
        #     return 0
        mean, _, var = gp.predictScalarLib(x)
        # mean, _, var = predictCholeskyScalar(x, Xm, Ym)
        std = np.sqrt(var)
        penalty = 0
        # if x[0] != int(x[0]):
        #         #     penalty = -9999
        #result = np.matrix(mean + beta * std) + penalty
        rX = x.copy()
        for k in discrete_idx:
            if spaces[k].isUniformlySpaced == False:
                rX[k] = min(int(round(spaces[k].getFloor(x[k]))), int(round(spaces[k].getCeil(x[k]))))
            else:
                rX[k] = int(round(x[k]))
        for k in cat_idx:
            if spaces[k].isUniformlySpaced == False:
                rX[k] = min(int(round(spaces[k].getFloor(x[k]))), int(round(spaces[k].getCeil(x[k]))))
            else:
                rX[k] = int(round(x[k]))
        meanRX, _, varRX = gp.predictScalarLib(rX)
        stdRX = np.sqrt(varRX)
        # if varRX == 0:
        #     return 0

        penalty = np.linalg.norm(np.array(rX) - np.array(x))
        inBeta = inEps - penalty
        # inBeta = inEps
        #print("Penalty:",penalty)
        # result = np.matrix(meanRX + inBeta * varRX)
        result = np.matrix(meanRX + inBeta * stdRX)
        return result

    @staticmethod
    def _ucb_d2(x, gp, beta):
        # if gp.checkExistInObs(x):
        #     return 0
        mean, _, var = gp.predictScalar(x)
        # mean, _, var = predictCholeskyScalar(x, Xm, Ym)
        var[var < 1e-10] = 0
        std = np.sqrt(var)
        result = np.matrix(mean + beta * std)
        return result

    @staticmethod
    def _lcb_d(x, gp, fmax, discrete_idx, beta):
        mean, _, var = gp.predictScalar(x)
        # mean, _, var = predictCholeskyScalar(x, Xm, Ym)
        var[var < 1e-10] = 0
        std = np.sqrt(var)
        inBeta = beta
        addPenalty = 0
        for i in discrete_idx:
            if x[i] != int(x[i]):
                floor = math.floor(x[i])
                ceil = math.ceil(x[i])
                mid = (ceil - floor) / 2
                penalty = (0.5 - abs((x[i] - floor) - mid))

                addPenalty = addPenalty + min(abs(x[i] - floor), abs(x[i] - ceil))
                inBeta = inBeta - addPenalty
            else:
                floor = math.floor(x[i])
                ceil = math.ceil(x[i])
                mid = (ceil - floor) / 2
                addPenalty = min(abs(x[i] - floor), abs(x[i] - ceil))
        result = np.matrix(mean - beta * std)
        return result

    @staticmethod
    def _lcb(x, gp, beta):
        mean, _, var = gp.predictScalar(x)
        # mean, _, var = predictCholeskyScalar(x, Xm, Ym)
        var[var < 1e-10] = 0
        std = np.sqrt(var)
        result = np.matrix(mean - beta * std)
        return result

    @staticmethod
    def _tpe(x, kdegood, kdebad, gamma=0.2):
        lx = kdegood.pdf(x)

        if lx == 0:
            return 1e-8
        prob = gamma + (kdebad.pdf(x) / lx) * (1 - gamma)
        result = 1.0/prob
        # result = lx/kdebad.pdf(x)
        return result

    @staticmethod
    def _pso(x, gp, fmax, kdegood, kdebad, epsilon, gamma=0.2):
        result1 = 0
        if(kdegood is not None):
            lx = kdegood.pdf(x)
            if lx == 0:
                return 1e-8
            prob = gamma + (kdebad.pdf(x) / lx) * (1 - gamma)
            result1 = 1.0/prob
        mean, _, var = gp.predictScalar(x)
        var2 = np.maximum(var, 1e-4 + 0 * var)
        std = np.sqrt(var2)
        Z = (mean - fmax - epsilon) / (std)
        result2 = (mean - fmax - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)

        return result2

    @staticmethod
    def _ei_d(x, gp, fMax, discrete_idx, spaces, epsilon, currentEps, stillGood, isSame, obs):
        # if x in np.array(obs):
        #     return 9999
        #mean, _, var = gp.predictScalar(x)
        # rX = x.copy()
        # for k in discrete_idx:
        #     if spaces[k].isUniformlySpaced == False:
        #         rX[k] = min(int(round(spaces[k].getFloor(x[k]))), int(round(spaces[k].getCeil(x[k]))))
        #     else:
        #         rX[k] = int(round(x[k]))
        # meanRX, _, varRX = gp.predictScalarLib(x)
        # stdRX = np.sqrt(varRX)
        # if(varRX == 0):
        #     return 0
        mean, _, var = gp.predictScalarLib(x)
        # mean, _, var = gp.predictScalar(x)
        #var2 = np.maximum(var, 0*1e-4 + 0 * var)
        # var[var < 1e-10] = 0
        std = np.sqrt(var)
        # if std == 0:
        #     return 0
        #(2) Rounding inside - uncomment if use
        # try:
        #     x = int(x)
        # except:
        #     for xi in x:
        #         xi = int(xi)
        #x[0] = int(x[0])
        actualEpsilon = epsilon

        # floorPoint = x.copy()
        # ceilPoint = x.copy()
        # sumPx = 0
        # for i in discrete_idx:
        #     if spaces[i].isUniformlySpaced == False:
        #         floorPoint[i] = spaces[i].getFloor(floorPoint[i])
        #         ceilPoint[i] = spaces[i].getCeil(ceilPoint[i])
        #         px = min(abs(x[i] - floorPoint[i]), abs(x[i] - ceilPoint[i]))
        #     else:
        #         floorPoint[i] = math.floor(floorPoint[i])
        #         ceilPoint[i] = math.ceil(ceilPoint[i])
        #         px = min(abs(x[i] - floorPoint[i]), abs(x[i] - ceilPoint[i]))
        #     sumPx += px
        # #
        # sumPx = 0
        # for i in discrete_idx:
        #     if spaces[i].isUniformlySpaced == False:
        #         px = min(abs(x[i] - spaces[i].getFloor(x[i])), abs(x[i] - spaces[i].getCeil(x[i])))
        #     else:
        #         px = min(abs(x[i] - math.floor(x[i])), abs(x[i] - math.ceil(x[i])))
        #     sumPx += px
        # # px = min(np.linalg.norm(np.array(x), np.array(floorPoint)), np.linalg.norm(np.array(x), np.array(ceilPoint)))
        # actualEpsilon = actualEpsilon + sumPx
        # print("actualEps:",actualEpsilon)
        #actualEpsilon = 0.01
        # normNear = gp.calculateNearestPoint(x)
        # currentEps = currentEps - (-1.0*normNear)
        # if stillGood:
        #     currentEps = currentEps
        # else:
        #     currentEps = currentEps/2
        # actualEpsilon = currentEps
        # floorPoint = x.copy()
        # ceilPoint = x.copy()
        # for i in discrete_idx:
        #     if spaces[i].isUniformlySpaced == False:
        #         floor = spaces[i].getFloor(x[i])
        #         ceil = spaces[i].getCeil(x[i])
        #         actualEpsilon = actualEpsilon + min(abs(x[i] - floor), abs(x[i] - ceil)) * inLambda
        #     else:
        #         floor = math.floor(x[i])
        #         ceil = math.ceil(x[i])
        #         px = min(abs(x[i] - floor), abs(x[i] - ceil))
        #         actualEpsilon = actualEpsilon + px * inLambda
            # floorPoint[i] = floor
            # ceilPoint[i] = ceil

        # floorPoint = x.copy()
        # ceilPoint = x.copy()
        # for i in discrete_idx:
        #     floor = math.floor(x[i])
        #     ceil = math.ceil(x[i])
        #     floorPoint[i] = floor
        #     ceilPoint[i] = ceil
        # px = min(np.linalg.norm(np.array(x) - np.array(floorPoint)), np.linalg.norm(np.array(x) - np.array(ceilPoint)))
        # actualEpsilon = actualEpsilon + px * 1.0

        #actualEpsilon = actualEpsilon + px * epsilon

        # if np.linalg.norm(np.array(x) - np.array(floorPoint)) != 0.0 and np.linalg.norm(np.array(x) - np.array(ceilPoint)) != 0.0:
        #     return 0


        # if type(floorPoint) != type([]):
        #     floorPoint = floorPoint.tolist()
        # if type(ceilPoint) != type([]):
        #     ceilPoint = ceilPoint.tolist()

        # # # print("floor:",type(floorPoint))
        # # # print("obs:", type(obs[0]))
        # if floorPoint in obs and ceilPoint in obs:
        #     mean, _, var = gp.predictScalarLib(ceilPoint)
        #     var2 = np.maximum(var, 1e-16 + 0 * var)
        #     std = np.sqrt(var2)
        #     Z = (mean - fMax - 0.01) / (std)
        #     result = (mean - fMax - actualEpsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        #     return result[0]

        # actualEpsilon = 0.01
        # _, _, varF = gp.predictScalarLib(floorPoint)
        # _, _, varC = gp.predictScalarLib(ceilPoint)
        # if varF == 0 and varC == 0:
        #     return 0
        # for i in discrete_idx:
        #     x[i] = int(x[i])

        # print("mean: ", mean)
        # print("var: ", var)
        #print("normNear:",normNear)

        #mean, _, var = gp.predictCholeskyScalar(x)
        #actualEpsilon = 0.01
        #print("mean:", mean)

        # result = np.matrix((mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + np.multiply(np.matrix(std.ravel()), (
        #     np.matrix(scipy.stats.norm.pdf(Z)))).ravel())
        # flag = False
        # for i in discrete_idx:
        #     if x[i] != int(x[i]):
        #         flag = True
        #         break
        # if flag == True:
        #     #result = (mean - fMax - actualEpsilon) * scipy.stats.norm.cdf(Z)
        #     result = 0
        # else:
        #     result = (mean - fMax - actualEpsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)

        # ZR = (meanRX - fMax - actualEpsilon) / (stdRX)
        # resultRX = (meanRX - fMax - actualEpsilon) * scipy.stats.norm.cdf(ZR) + stdRX * scipy.stats.norm.pdf(ZR)
        Z = (mean - fMax - actualEpsilon) / (std)
        result = (mean - fMax - actualEpsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)

        #print("x inside:",x)
        #isFloat = False
        # for xi in x:
        #     if xi != int(xi):
        #         return -99999
        #         # isFloat = True
        #         # break

        # if isFloat:
            #print("x inside:", x, " is float")
            # return -9999
            #raise Exception("Invalid input")

            # Xfloor = np.array([int(np.floor(xtmp)) for xtmp in x])
            # Xceil = np.array([int(np.ceil(xtmp)) for xtmp in x])
            # left = _inter_ei(Xfloor)
            # right = _inter_ei(Xceil)
            # slope = (right - left) / np.linalg.norm(Xceil - Xfloor)
            # # print("slope:",slope)
            # result = _inter_ei(Xfloor) + np.linalg.norm(x - Xfloor) * slope
        # if x[0] != int(x[0]):
        #     return -99999

        # floorPoint = x.copy()
        # ceilPoint = x.copy()
        # sumPx = 0
        # for i in discrete_idx:
        #     if spaces[i].isUniformlySpaced == False:
        #         floorPoint[i] = spaces[i].getFloor(floorPoint[i])
        #         ceilPoint[i] = spaces[i].getCeil(ceilPoint[i])
        #     else:
        #         floorPoint[i] = math.floor(floorPoint[i])
        #         ceilPoint[i] = math.ceil(ceilPoint[i])

        # if(np.linalg.norm(np.array(x) - np.array(floorPoint)) != 0 and np.linalg.norm(np.array(x) - np.array(ceilPoint)) != 0):
        #     slope = (np.array(floorPoint) - np.array(ceilPoint)) / np.linalg.norm(np.array(ceilPoint) - np.array(floorPoint))
        #
        #     meanF, _, varF = gp.predictScalarLib(floorPoint)
        #     meanC, _, varC = gp.predictScalarLib(ceilPoint)
        #     stdF = np.sqrt(varF)
        #     stdC = np.sqrt(varC)
        #     ZF = (meanF - fMax - actualEpsilon) / (stdF)
        #     ZC = (meanC - fMax - actualEpsilon) / (stdC)
        #     resultF = (meanF - fMax - actualEpsilon) * scipy.stats.norm.cdf(ZF) + stdF * scipy.stats.norm.pdf(ZF)
        #     resultC = (meanC - fMax - actualEpsilon) * scipy.stats.norm.cdf(ZC) + stdC * scipy.stats.norm.pdf(ZC)
        #     return resultF + np.linalg.norm(np.array(x) - np.array(floorPoint)) * slope



        #return result[0] - sumPx
        return result[0]

    @staticmethod
    def _ei_new(x, gp, fMax, discrete_idx, cat_idx, dens_good, dens_bad, spaces, epsilon, currentEps, stillGood, isSame, obs):
        # rX = x.copy()
        # for k in discrete_idx:
        #     if spaces[k].isUniformlySpaced == False:
        #         rX[k] = min(int(round(spaces[k].getFloor(x[k]))), int(round(spaces[k].getCeil(x[k]))))
        #     else:
        #         rX[k] = int(round(x[k]))
        # for k in cat_idx:
        #     if spaces[k].isUniformlySpaced == False:
        #         rX[k] = min(int(round(spaces[k].getFloor(x[k]))), int(round(spaces[k].getCeil(x[k]))))
        #     else:
        #         rX[k] = int(round(x[k]))
        # meanRX, _, varRX = gp.predictScalarLib(rX)
        # stdRX = np.sqrt(varRX)
        # x = rX
        #
        # catProb = 0
        #
        # if dens_good != None:
        #     for k in cat_idx:
        #         #print("rX[k]:", rX[k])
        #         if dens_good.pdf([rX[k]]) != 0:
        #             #catProb += dens_good.pdf([rX[k]])/dens_bad.pdf([rX[k]])
        #             catProb += dens_good.pdf([rX[k]])
        #             #catProb += dens_bad.pdf([rX[k]])/dens_good.pdf([rX[k]])

        # floorPoint = x.copy()
        # ceilPoint = x.copy()
        # sumPx = 0
        # for i in discrete_idx:
        #     if spaces[i].isUniformlySpaced == False:
        #         floorPoint[i] = spaces[i].getFloor(floorPoint[i])
        #         ceilPoint[i] = spaces[i].getCeil(ceilPoint[i])
        #         px = min(abs(x[i] - floorPoint[i]), abs(x[i] - ceilPoint[i]))
        #     else:
        #         floorPoint[i] = math.floor(floorPoint[i])
        #         ceilPoint[i] = math.ceil(ceilPoint[i])
        #         px = min(abs(x[i] - floorPoint[i]), abs(x[i] - ceilPoint[i]))
        #     sumPx += px

        mean, _, var = gp.predictScalarLib(x)
        #var2 = np.maximum(var, 0*1e-4 + 0 * var)
        # var[var < 1e-10] = 0
        std = np.sqrt(var)


        actualEpsilon = epsilon
        #print("",actualEpsilon)
        # # Z = (mean - fMax - actualEpsilon) / (std)
        # # result = (mean - fMax - actualEpsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        Z = (mean - fMax - actualEpsilon) / (std)
        result = (mean - fMax - actualEpsilon) * scipy.stats.norm.cdf(Z) + (std) * scipy.stats.norm.pdf(Z)

        # ZRX = (meanRX - fMax - actualEpsilon) / (stdRX)
        # resultRX = (meanRX - fMax - actualEpsilon) * scipy.stats.norm.cdf(ZRX) + stdRX * scipy.stats.norm.pdf(ZRX)

        return result[0]

    @staticmethod
    def _eiTest(x, gp, fMax, discrete_idx, spaces, epsilon, currentEps, stillGood, isSame, obs):
        mean, _, var = gp.predictScalarLib(x)
        std = np.sqrt(var)

        # actualEpsilon = epsilon
        actualEpsilon = 0.01

        floorPoint = x.copy()
        ceilPoint = x.copy()
        sumPx = 0
        for i in discrete_idx:
            if spaces[i].isUniformlySpaced == False:
                floorPoint[i] = spaces[i].getFloor(floorPoint[i])
                ceilPoint[i] = spaces[i].getCeil(ceilPoint[i])
                px = min(abs(x[i] - floorPoint[i] ), abs(x[i] - ceilPoint[i]))
            else:
                floorPoint[i] = math.floor(floorPoint[i])
                ceilPoint[i] = math.ceil(ceilPoint[i])
                px = min(abs(x[i] - floorPoint[i]), abs(x[i] - ceilPoint[i]))
            sumPx += px

        #px = min(np.linalg.norm(np.array(x), np.array(floorPoint)), np.linalg.norm(np.array(x), np.array(ceilPoint)))
        actualEpsilon = actualEpsilon + sumPx
        # # floorPoint = x.copy()
        # # ceilPoint = x.copy()
        # sumPx = 0
        # for i in discrete_idx:
        #     if spaces[i].isUniformlySpaced == False:
        #         floor = spaces[i].getFloor(x[i])
        #         ceil = spaces[i].getCeil(x[i])
        #         px = min(abs(x[i] - floor), abs(x[i] - ceil))
        #         #actualEpsilon = actualEpsilon + min(abs(x[i] - floor), abs(x[i] - ceil)) * lamda
        #         sumPx += px * lamda
        #     else:
        #         floor = math.floor(x[i])
        #         ceil = math.ceil(x[i])
        #         px = min(abs(x[i] - floor), abs(x[i] - ceil))
        #         #print("actualEps:", actualEpsilon, " px:", px, " lamda:",lamda)
        #         #actualEpsilon = actualEpsilon + px * lamda
        #         sumPx += px * lamda
        # actualEpsilon = actualEpsilon + sumPx

        Z = (mean - fMax - actualEpsilon) / (std)

        result = (mean - fMax - actualEpsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        return result[0]

    @staticmethod
    def _eiRound(x, gp, fMax, discrete_idx, spaces, epsilon, currentEps, stillGood, isSame, obs):
        rX = x.copy()
        for k in discrete_idx:
            if spaces[k].isUniformlySpaced == False:
                rX[k] = min(int(round(spaces[k].getFloor(x[k]))), int(round(spaces[k].getCeil(x[k]))))
            else:
                rX[k] = int(round(x[k]))
        mean, _, var = gp.predictScalarLib(rX)
        std = np.sqrt(var)

        # actualEpsilon = epsilon
        actualEpsilon = 0.01

        Z = (mean - fMax - actualEpsilon) / (std)

        result = (mean - fMax - actualEpsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        return result[0]

    @staticmethod
    def _ucb_optv(x, gp, dim, ite, inEps):
        #ite from 1 to n
        # mean, _, var = gp.predictScalar(x)
        # mean, _, var = predictCholeskyScalar(x, Xm, Ym)
        mean, _, var = gp.predictScalarLib(x)
        std = np.sqrt(var)
        penalty = 0
        # if x[0] != int(x[0]):
        #         #     penalty = -9999
        # result = np.matrix(mean + beta * std) + penalty
        tau = 2.0 * np.log(pow(ite, dim/2.0 + 2.0) * math.pi**2/(3*0.1))
        v = inEps
        inBeta = math.sqrt(v*tau)

        # print("Penalty:",penalty)
        result = (mean + inBeta * std)
        return result

    @staticmethod
    def _gaussian(x, mean, sigma=0.1):
        res = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) ** 2) / (sigma ** 2))
        return res

    @staticmethod
    def _ucb_opteta(x, gp, dim, ite, inEps, betaT, discrete_idx, cat_idx, spaces, obs, sigma_dup=0.01):
        # ite from 1 to n
        # mean, _, var = gp.predictScalar(x)
        # mean, _, var = predictCholeskyScalar(x, Xm, Ym)

        mean, _, var = gp.predictScalarLib(x)
        std = np.sqrt(var)

        # rX = x.copy()
        # for k in discrete_idx:
        #     if spaces[k].isUniformlySpaced == False:
        #         rX[k] = min(int(round(spaces[k].getFloor(x[k]))), int(round(spaces[k].getCeil(x[k]))))
        #     else:
        #         rX[k] = int(round(x[k]))
        # for k in cat_idx:
        #     if spaces[k].isUniformlySpaced == False:
        #         rX[k] = min(int(round(spaces[k].getFloor(x[k]))), int(round(spaces[k].getCeil(x[k]))))
        #     else:
        #         rX[k] = int(round(x[k]))
        # # meanRX, _, varRX = gp.predictScalarLib(rX)
        # # #stdRX = np.sqrt(varRX)
        # #
        # penalty = np.linalg.norm(np.array(rX) - np.array(x))
        # if x[0] != int(x[0]):
        #         #     penalty = -9999
        # result = np.matrix(mean + beta * std) + penalty
        # print(x)

        # rX = x.copy()
        # # ceilPoint = x.copy()
        # # floorPoint = x.copy()
        # for k in discrete_idx:
        #     # ceilPoint[k] = spaces[k].getCeil(x[k])
        #     # floorPoint[k] = spaces[k].getFloor(x[k])
        #     if spaces[k].isUniformlySpaced == False:
        #         if abs(rX[k] - spaces[k].getCeil(rX[k])) > abs(
        #                 rX[k] - spaces[k].getFloor(rX[k])):
        #             rX[k] = spaces[k].getFloor(rX[k])
        #         else:
        #             rX[k] = spaces[k].getCeil(rX[k])
        #         # rX[k] = min(int(round(spaces[k].getFloor(x[k]))), int(round(spaces[k].getCeil(x[k]))))
        #     else:
        #         rX[k] = int(round(x[k]))
        #         # rX[k] = int(np.floor(x[k]))
        # for k in cat_idx:
        #     if spaces[k].isUniformlySpaced == False:
        #         if abs(rX[k] - spaces[k].getCeil(rX[k])) > abs(
        #                 rX[k] - spaces[k].getFloor(rX[k])):
        #             rX[k] = spaces[k].getFloor(rX[k])
        #         else:
        #             rX[k] = spaces[k].getCeil(rX[k])
        #     else:
        #         rX[k] = int(round(x[k]))
        # meanRX, _, varRX = gp.predictScalarLib(rX)
        # stdRX = np.sqrt(varRX)

        # meanC, _, varC = gp.predictScalarLib(ceilPoint)
        # stdC = np.sqrt(varC)
        # meanF, _, varF = gp.predictScalarLib(floorPoint)
        # stdF = np.sqrt(varF)
        #
        # inBeta = inEps
        # resC = (meanC + inBeta * stdC)
        # resF = (meanF + inBeta * stdF)
        # # print(x," ", ceilPoint, " ", floorPoint)
        # verticalChange = resC - resF
        # horizontalChange = np.linalg.norm(ceilPoint-floorPoint)
        # m = 1.0*verticalChange/horizontalChange
        # result = resC - m*(np.linalg.norm(ceilPoint - x))

        # eta = 0
        # for xo in obs:
        #     #flag = False
        #     # vdx = []
        #     # vdrx = []
        #     # for k in discrete_idx:
        #     #     vdx.append(xo[k])
        #     #     # vdx.append(rX[k])
        #     #     vdrx.append(x[k])
        #         # if spaces[k].isUniformlySpaced == False:
        #         #     if abs(x[k] - spaces[k].getCeil(x[k])) > 0 and abs(x[k] - spaces[k].getFloor(x[k])) > 0:
        #         #         vdrx.append(spaces[k].getFloor(x[k]))
        #         #     else:
        #         #         vdrx.append(spaces[k].getCeil(x[k]))
        #         # else:
        #         #     if abs(x[k] - int(round(x[k]))) > 0:
        #         #         vdrx.append(int(round(x[k])))
        #     # print("x:",x, " xo:", xo, " norm:",np.linalg.norm(np.array(vdx) - np.array(vdrx)))
        #     # print("vdrx:", vdrx, " vdx:", vdx, " x:", x, " xo:", xo)
        #     # if np.linalg.norm(np.array(vdrx) - np.array(vdx)) > 0.0:
        #     #     eta += np.exp(-0.5 * (np.linalg.norm(x - xo) ** 2) / (0.5 ** 2)) #0.3 branin #Current 0.1
        #     eta += np.exp(-0.5 * (np.linalg.norm(x - xo) ** 2) / (0.1 ** 2))  # 0.3 branin, 0.01 mixed branin
        # #
        # eta = 1 - eta
        # # # eta = (len(obs) - eta)/len(obs)
        # eta = 1

        # # if flag:
        # # eta = 0
        # # for xo in obs:
        # #     # eta += 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (np.linalg.norm(x - xo) ** 2) / (sigma ** 2))
        # #     # eta += np.exp(-0.5 * (np.linalg.norm(x - xo) ** 2) / (0.001 ** 2))
        # #     #eta += np.exp(-0.5 * (np.linalg.norm(x - xo) ** 2) / (0.5 ** 2))
        # #     eta += np.exp(-0.5 * (np.linalg.norm(x - xo) ** 2) / (0.3 ** 2))
        # # #eta = (len(obs) - eta)/len(obs)
        # # eta = 1 - eta
        # g = np.exp(-0.5 * (np.linalg.norm(x - rX) ** 2) / (0.42 ** 2)) #current 0.01 #5d grie 0.1 and 0.5
        # # g = np.exp(-0.5 * (np.linalg.norm(x - rX) ** 2) / (0.01 ** 2)) #Mixed branin
        # # print("x:", x, " rX:", rX, " norm:",np.linalg.norm(rX - rX))
        # #mixed branin 0.01 and 0.01
        # # penalty = np.linalg.norm(np.array(rX) - np.array(x))
        # # print("x:", x, " eta:",eta)
        # # print(g)
        inBeta = inEps
        # # eta = 1
        # g = 1
        # # inBeta = betaT
        # # print("Penalty:",penalty)
        # # result = (mean + inBeta * std) * eta#+ penalty

        result = (mean + inBeta * std) * 1 * 1# + penalty
        # result = (meanRX + inBeta * std)# + penalty
        # result = (meanRX + inBeta * stdRX)
        # result = (mean + inBeta * std)
        # result = (meanRX + inBeta * stdRX) * eta * g  # + penalty
        # result = (mean + inBeta * std) - (1 - g)*std  # + penalty
        # result = (meanRX + inBeta * stdRX)# + penalty
        # result = (meanRX + inBeta * stdRX) * eta - (1 - g)
        # result = (meanRX + inBeta * stdRX)
        # result = eta
        return result

