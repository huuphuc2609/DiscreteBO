import GPy
import numpy as np
from GPy import Param
from paramz.transformations import Logexp

from core.BO.kernels import Kernels
from scipy.linalg import cho_solve, cholesky, solve_triangular

class GaussianProcess():
    def __init__(self,kernelName, numDim, func, isTransformation):
        self.kernel = Kernels(kernelName)
        self.noise_delta = 1e-6
        # self.noise_delta = 0.01

        self.Xm = []
        self.XmTran = []
        self.Ym = []

        self.kxx = []
        self.kXmXm = []
        self.kXmXm_inv = []
        self.kxXm = []
        self.discrete_idx = []

        self.input_dim = numDim
        self.func = func
        self.isTran = isTransformation

        # if isTransformation == True:
        #     self.libKern = GPy.kern.Matern52_Round(self.input_dim, func, variance=0.1, ARD=False)  # + GPy.kern.Bias(self.input_dim)
        #     self.libKern = GPy.kern.GridRBF(self.input_dim, variance=1.0, ARD=False)  # + GPy.kern.Bias(self.input_dim)
        # else:
        #     #self.libKern = GPy.kern.Matern52(self.input_dim, variance=0.1, ARD=False)  # + GPy.kern.Bias(self.input_dim)
        #self.libKern = GPy.kern.GridRBF(self.input_dim, variance=1.0, ARD=False)  # + GPy.kern.Bias(self.input_dim)
        self.libKern = GPy.kern.RBF(self.input_dim, lengthscale=0.1, ARD=True)  # + GPy.kern.Bias(self.input_dim)
        self.optimizeLS = True
        # self.libKern = GPy.kern.Matern52(self.input_dim, ARD=False)
        # self.libKern = GPy.kern.Matern52(self.input_dim, variance=0.1, ARD=False)

        self.libOptimizer = 'bfgs'
        self.model = None

    def checkExistInObs(self, inputX):
        tmp = np.array(inputX)
        obs = np.array(self.Xm)
        if tmp in obs:
            return True
        else:
            return False

    def getLengthScale(self):
        res = self.model.kern.lengthscale.values.tolist()
        return res[0]

    def setLengthScale(self, len):
        # self.libKern = GPy.kern.RBF(self.input_dim, self.func, self.isTran, lengthscale=len, ARD=False)
        self.libKern.lengthscale = Param('lengthscale', len, Logexp())


    def setOptimizeLS(self, val):
        self.optimizeLS = val

    def setDiscreteIdx(self, discrete_idx):
        self.discrete_idx = discrete_idx

    def setCategoricalIdx(self, categorical_idx):
        self.categorical_idx = categorical_idx

    def fit(self,Xmeasurement,Ymeasurement):
        self.Xm = Xmeasurement
        self.Ym = Ymeasurement

        # if self.isTran:
        #     self.XmTran = []
        #     for tmpX in self.Xm:
        #         self.XmTran.append(self.roundX(tmpX))
        #     #self.Xm = self.XmTran

        #Use lib
        if self.model is None:
            add_noise = self.noise_delta
            innoise_var = np.array(Ymeasurement).var() * add_noise
            #innoise_var = np.array(Ymeasurement).var() * 0
            self.model = GPy.models.GPRegression(np.array(self.Xm), np.array(Ymeasurement).reshape(-1,1), self.libKern, noise_var=innoise_var, normalizer=True)
            self.model.Gaussian_noise.constrain_fixed(add_noise, warning=False)
        else:
            # print("Xm:",self.Xm)
            # print("Ymeasurement:", Ymeasurement)
            self.model.set_XY(np.array(self.Xm), np.array(Ymeasurement).reshape(-1,1))
        #self.model.optimize(optimizer=self.libOptimizer, max_iters=1000, messages=False, ipython_notebook=False)
        if self.optimizeLS:
            try:
                self.model.optimize_restarts(5, verbose=False)
            except np.linalg.linalg.LinAlgError:
                pass



        # if self.kernel.name == "matern":
        #     self.kernel.amp = 1.5
        #     bnds = ((1e-5, 1e5),)  # Define boundary tuple
        #     # nu in [0.5, 1.5, 2.5, inf]
        #     #PSO
        #     lb = []
        #     ub = []
        #     for i in bnds:
        #         lb.append(i[0])
        #         ub.append(i[1])
        #     PSO = ParticleSwarmOptimizer()
        #     optimized, _ = PSO.minimize(lambda x: -self.log_marginal_likelihood(x), lb, ub,
        #                                 f_ieqcons=None, maxiter=40, swarmsize=10)
        #     self.kernel.lengthScale = optimized[0]
        #     #self.kernel.amp = optimized[1]
        #     #L-BFGS-B
        #     # resvals = minimize(lambda x: -self.log_marginal_likelihood(x), [1.0], bounds=bnds,
        #     #                    method="L-BFGS-B")
        #     # optimized = (resvals.x[:]).tolist()
        #     # self.kernel.lengthScale = optimized[0]
        #     #DIRECT
        #     # b = np.array(bnds)
        #     # # DIRECTstart_time = time.time()
        #     # DIRECTOptimizer = DIRECTAlgo()
        #     # # print("Log before:", self.log_marginal_likelihood([1.0, 1.0]))
        #     # resvals, nextY, _, fc = DIRECTOptimizer.minimize(lambda x: -self.log_marginal_likelihood(x),
        #     #                                                  b, max_iters=200, max_evals=1200, max_deep=1400)
        #     #print("resvals:",resvals)
        #     # self.kernel.lengthScale = resvals
        #     # self.kernel.lengthScale = 1.0
        #     # self.kernel.amp = 1.5
        #     # self.kernel.noise_delta = 10e-8
        # else:
        #     bnds = ((0.01, 10), (0.01, 10),)  # Define boundary tuple
        #     # nu in [0.5, 1.5, 2.5, inf]
        #     # PSO
        #     lb = []
        #     ub = []
        #     for i in bnds:
        #         lb.append(i[0])
        #         ub.append(i[1])
        #
        #     def obj_der(x):
        #         return nd.Jacobian(lambda x: -self.log_marginal_likelihood(x))(x)[0]
        #
        #     def obj_hess(x):
        #         return nd.Hessian(lambda x: -self.log_marginal_likelihood(x))(x)[0]
        #
        #     #linear_constraint = LinearConstraint([[1, 1]], [0, 0], [100, 100])
        #     optimizeEps = minimize(lambda x: -self.log_marginal_likelihood(x),
        #                            #[np.random.uniform(leftBetaBound, rightBetaBound)], method='trust-ncg',
        #                            # [0.0], method='trust-ncg',
        #                            #[1.0, 1.0], method='trust-constr',
        #                            [1.0, 1.0], method='BFGS',
        #                            #[1.0, 1.0], method='CG',
        #                            #constraints=[linear_constraint],
        #                            jac=obj_der, hess=obj_hess,
        #                            #bounds=bndsEps,
        #                             options={'xtol': 1e-8, 'disp': False})
        #     optimized = (optimizeEps.x[:]).tolist()
        #     #print(optimized)
        #     # PSO = ParticleSwarmOptimizer()
        #     # optimized, _ = PSO.minimize(lambda x: -self.log_marginal_likelihood(x), lb, ub,
        #     #                             f_ieqcons=None, maxiter=40, swarmsize=10)
        #     self.kernel.lengthScale = optimized[0]
        #     self.kernel.amp = optimized[1]
        #     # self.kernel.lengthScale = 1.0
        #     # self.kernel.amp = 1.0
        #     #DIRECT
        #     # b = np.array(bnds)
        #     # # DIRECTstart_time = time.time()
        #     # DIRECTOptimizer = DIRECTAlgo()
        #     # #print("Log before:", self.log_marginal_likelihood([1.0, 1.0]))
        #     # resvals, nextY, _, fc = DIRECTOptimizer.minimize(lambda x: -self.log_marginal_likelihood(x),
        #     #                                            b, max_iters=200, max_evals=1200, max_deep=1400)
        #     # self.kernel.lengthScale = resvals[0]
        #     # self.kernel.amp = resvals[1]
        #     #L-BFGS-B
        #     # resvals = minimize(lambda x: -self.log_marginal_likelihood(x), [1.0, 1.0], bounds=bnds,
        #     #                    method="L-BFGS-B")
        #     # optimized = (resvals.x[:]).tolist()
        #     # self.kernel.lengthScale = optimized[0]
        #     # self.kernel.amp = optimized[1]
        #
        #     #print("Log After:", self.log_marginal_likelihood([resvals[0], resvals[1]]))
        #     # self.kernel.lengthScale = 1.0
        #     # self.kernel.amp = 1.0
        #     # self.kernel.noise_delta = 10e-8

    def log_marginal_likelihood(self,input):
        self.kernel.lengthScale = input[0]
        if self.kernel.name != "matern":
            self.kernel.amp = input[1]
        K = self.kernel.k(self.Xm, self.Xm) + self.noise_delta * np.eye(len(self.Xm))
        K_inv = np.linalg.pinv(K)
        Ymm = np.array(self.Ym)
        log_likelihood = -0.5 * Ymm.T.dot(K_inv).dot(Ymm)
        log_likelihood -= 0.5 * np.log(np.linalg.det(K))
        log_likelihood -= K.shape[0] / 2 * np.log(2 * np.pi)
        return np.asscalar(log_likelihood)

    def calculateNearestPoint(self,X):
        self.dL = []
        for i in self.Xm:
            # print("Xmi:",i)
            # print("X:", X)
            norm = abs(np.linalg.norm((np.array(X)-np.array(i))))
            self.dL.append(norm)
        #print("dL: ",self.dL)
        return self.Ym[np.argmax(self.dL[0])]

    def predict(self, x):
        self.kxx = self.kernel.k(x, x) + np.diag(np.eye(len(x))*self.noise_delta)
        self.kXmXm = self.kernel.k(self.Xm, self.Xm)

        self.kXmXm_inv = np.linalg.pinv(self.kXmXm)

        self.kxXm = self.kernel.k(x, self.Xm) + self.noise_delta

        mu_s = np.dot(np.dot(self.kxXm, self.kXmXm_inv), self.Ym)
        cov_s = self.kxx - np.dot(np.dot(self.kxXm, self.kXmXm_inv), np.transpose(self.kxXm))
        var_s = np.absolute(np.matrix(np.diag(np.matrix(cov_s))))
        var_s = np.clip(var_s, 1e-10, np.inf)
        return mu_s, cov_s, var_s

    def predictScalar(self, x):
        x = np.array(x).reshape(1, -1)

        if self.isTran:
            mu_s, cov_s, var_s = self.predictScalarTrans(x)
            return mu_s, cov_s, var_s

        self.kxx = self.kernel.k(x, x) + self.noise_delta
        self.kXmXm = self.kernel.k(self.Xm, self.Xm) + self.noise_delta
        self.kXmXm_inv = np.linalg.pinv(self.kXmXm)

        self.kxXm = self.kernel.k(x, self.Xm) + self.noise_delta

        mu_s = np.dot(np.dot(self.kxXm, self.kXmXm_inv), self.Ym)
        cov_s = self.kxx - np.dot(np.dot(self.kxXm, self.kXmXm_inv), np.transpose(self.kxXm))
        var_s = np.absolute(np.matrix(np.diag(np.matrix(cov_s))))
        var_s = np.clip(var_s, 1e-10, np.inf)

        return mu_s, cov_s, var_s

    def predictScalarLib(self, x):
        if self.isTran:
            x = self.roundX(x)
        x = np.array(x).reshape(1, -1)
        #use lib
        #newModel = self.model.copy()
        # m, v = newModel.predict(x, full_cov=False, include_likelihood=False)
        m, v = self.model.predict(x, full_cov=False, include_likelihood=False)

        var_s = np.clip(v, 1e-10, np.inf)
        mu_s = m

        return mu_s, 0, var_s

    def roundX(self,x):
        xT = x.copy()
        for k in self.func.discreteIdx:
            if self.func.spaces[k].isUniformlySpaced == False:
                if abs(x[k] - self.func.spaces[k].getCeil(x[k])) > abs(
                        x[k] - self.func.spaces[k].getFloor(x[k])):
                    xT[k] = self.func.spaces[k].getFloor(x[k])
                else:
                    xT[k] = self.func.spaces[k].getCeil(x[k])
            else:
                xT[k] = int(np.round(x[k]))
        for k in self.func.categoricalIdx:
            if self.func.spaces[k].isUniformlySpaced == False:
                if abs(x[k] - self.func.spaces[k].getCeil(x[k])) > abs(
                        x[k] - self.func.spaces[k].getFloor(x[k])):
                    xT[k] = self.func.spaces[k].getFloor(x[k])
                else:
                    xT[k] = self.func.spaces[k].getCeil(x[k])
            else:
                xT[k] = int(np.round(x[k]))
        return xT

    def predictScalarTrans(self, x):
        xT = self.roundX(x)
        self.XmTran = []
        for tmpX in self.Xm:
            self.XmTran.append(self.roundX(tmpX))

        xT = np.array(xT).reshape(1, -1)
        self.kxx = self.kernel.k(xT, xT) + self.noise_delta
        self.kXmXm = self.kernel.k(self.XmTran, self.XmTran) + self.noise_delta
        self.kXmXm_inv = np.linalg.pinv(self.kXmXm)

        self.kxXm = self.kernel.k(xT, self.XmTran) + self.noise_delta

        mu_s = np.dot(np.dot(self.kxXm, self.kXmXm_inv), self.Ym)
        cov_s = self.kxx - np.dot(np.dot(self.kxXm, self.kXmXm_inv), np.transpose(self.kxXm))
        var_s = np.absolute(np.matrix(np.diag(np.matrix(cov_s))))
        var_s = np.clip(var_s, 1e-10, np.inf)
        return mu_s, cov_s, var_s

    def predictCholeskyScalar(self, x):
        x = np.array(x).reshape(1, -1)
        self.kXmXm = self.kernel.k(self.Xm, self.Xm)
        self.kxXm = self.kernel.k(x, self.Xm)
        L = cholesky(self.kXmXm, lower=True)
        alpha = cho_solve((L, True), self.Ym)
        # y_train_mean = np.mean(Y_query)
        mu_s = self.kxXm.dot(alpha)  # Line 4 (y_mean = f_star)
        # y_mean = y_train_mean + y_mean  # undo normal.

        ######################CovByCholesky######################
        v = cho_solve((L, True), self.kxXm.T)  # Line 5
        cov_s = self.kernel.k(x, x) - self.kxXm.dot(v)  # Line 6

        ######################StandardDeviationByCholesky######################
        # compute inverse K_inv of K based on its Cholesky
        # decomposition L and its inverse L_inv
        L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
        self.kXmXm_inv = L_inv.dot(L_inv.T)
        # Compute variance of predictive distribution
        y_var = np.matrix(np.diag(np.matrix(self.kernel.k(x, x)))) - np.einsum("ij,ij->i", np.dot(self.kxXm, self.kXmXm_inv),
                                                                               self.kxXm)
        # std_s = np.sqrt(np.absolute(y_var))
        return mu_s, cov_s, y_var