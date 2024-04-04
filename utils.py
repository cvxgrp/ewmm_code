from __future__ import annotations
import time
import numpy as np
import scipy as sp
import cvxpy as cp
from abc import ABC, abstractmethod
from tqdm import tqdm
import multiprocessing as mp

class NonStationaryProcess(ABC):
    @abstractmethod
    def gen(self, n):
        raise NotImplementedError

    
class RandomFourier(NonStationaryProcess):
    def __init__(self, d, T, k_fouier, sigma, S=1600):
        """
        Initialize the random sinonoidal coefficients
        """
        self.d = d
        self.sigma = sigma
        self.k_fouier = k_fouier

        self.W = np.random.rand(self.k_fouier) / self.k_fouier ** .5

        periods = np.zeros((self.d,self.k_fouier))
        for i in range(self.k_fouier):
            periods[:,i] = np.random.rand(self.d) * 10*S + S 
        self.periods = periods

        self.shift = np.random.rand(self.d, self.k_fouier) * periods

    def gen(self,T):
        """
        Generate T samples from the process
        """
        t = np.arange(T)[:,None,None] - self.shift
        theta_ = np.sin(2 * np.pi * t / self.periods)
        theta = theta_ @ self.W

        X = np.random.randn(T, self.d)
        y = (X * theta).sum(axis=1) + np.random.randn(T)*self.sigma
        y = (y >= 0).astype(int)

        return X, y, theta


class Evaluator:
    def __init__(self, models, X, y, thetas):
        """
        Class for evaluating the models on the data X, y, with thetas as the ground truth
        """
        self.models = models
        self.X = X
        self.y = y
        self.thetas = thetas

    def evaluate(self):
        """
        Evaluate the models on the data and return the results in a dictionary
        """
        results = {}
        for model in self.models:
            print("evaluating ", model.name)
            y_pred = model.online_pred(self.X, self.y)
            results[model.name] = y_pred

        if self.X is not None:
            ground_preds = (self.X * self.thetas).sum(axis=1)
            ground_preds = (ground_preds >= 0).astype(int)
            results["ground"] = ground_preds

        return results
    
class Model(ABC):
    @abstractmethod
    def online_pred(self, X, y):
        raise NotImplementedError
        
class Logistic_tail(Model):
    def __init__(self, H, gamma, alpha, name='Logistic_tail'):
        """
        Initialize the exponential weighted moving logistic regression model
        with tail approximation
        """
        self.name = name
        self.H = H
        self.gamma = gamma
        self.alpha = alpha

        
    def fit(self, X, y, theta_current, P, q, r):
        """
        Fit the model to the data X, y, with the current estimate of theta and
        the previous quadratic approximation to the tail
        """
        buffer_X = X[-self.H:]
        buffer_y = y[-self.H:]
        H_eff = len(buffer_X)
        weight = self.gamma ** np.arange(H_eff)[::-1]
        weight = weight / weight.sum()

        # create a cvxpy logistic regression problem using buffer X and y, with
        # a loss weighted by weight
        theta = cp.Variable(X.shape[1])
        likelihoods = cp.multiply(buffer_y, buffer_X @ theta) - cp.logistic(buffer_X @ theta)
        ll = cp.sum(cp.multiply(weight, likelihoods))

        # define l_final to be the loss on buffer_X[-1] and buffer_y[-1]
        x_final, y_final = buffer_X[-1], buffer_y[-1]
        l_final = y_final * (x_final @ theta_current) - np.log(1 + np.exp(x_final @ theta_current))

        # define H_final to be the Hessian of l_final wrt theta, the analytical
        # form of which is given below evaluated at theta_current
        # Some steps are taken to make this more numerically stable

        z = x_final @ theta_current
        s_2 = 2*(np.log(1+np.exp(-z))+z)
        s = z - s_2
        S = np.exp(s)
        H_final = -np.outer(x_final,x_final)*S

        # define g_final to be the gradient of l_final wrt theta, evaluated at
        # theta_current
        g_final = buffer_X[-1] * (buffer_y[-1] - 1 / (1 + np.exp(buffer_X[-1] @ theta_current)))

        if P is not None:
        # a_t/a_t-1 * beta on first term, a_t on second term
            t = len(X)
            w_1 = self.gamma * (1-(self.gamma ** (t-1)))/(1-(self.gamma ** t))
            w_2 = (1 - self.gamma) / (1 - (self.gamma ** t))

            P_new = w_1 * P + w_2 * H_final
            q_new = w_1 * q + w_2 * g_final
            r_new = w_1 * r + w_2 * l_final
        else:
            P_new = H_final
            q_new = g_final
            r_new = l_final

        reg = self.alpha * cp.sum_squares(theta)
        tail = cp.quad_form(theta - theta_current, P_new) + q_new.T @ (theta - theta_current) + r_new
        # tail = 0.0 # TODO: remove
        objective = ll + tail - reg
    
        # create the problem and solve it
        constraints = []
        prob = cp.Problem(cp.Maximize(objective), constraints)
        prob.solve(solver=cp.MOSEK)

        theta = theta.value
        return theta, P_new, q_new, r_new

    def online_pred(self, X, y):
        """
        Fit the model to the data X, y and return the predictions and times and thetas
        """
        preds = np.zeros(X.shape[0])
        times = np.zeros(y.shape[0])
        theta = np.zeros(X.shape[1])
        thetas = np.zeros((X.shape[0], X.shape[1]))

        P, q, r = None, None, None

        for i in tqdm(range(1,X.shape[0])):
            X_train = X[:i]
            y_train = y[:i]
            weight = self.gamma ** np.arange(i)[::-1]
            weight = weight / weight.sum()

            # fit the model
            ts = time.time()
            theta, P, q, r = self.fit(X_train, y_train, theta, P, q, r)
            te = time.time()

            thetas[i] = theta
            pred = theta @ X[i]
            # turn pred into 0-1 value
            pred = float(pred >= 0)

            preds[i] = pred
            times[i] = te - ts

        return preds, times, thetas
                
class QuantileEstimator(Model):
    def __init__(self, alpha, gamma, name='QuantileEstimator', tail_approx: None | int = None, grad=False):
        """
        Initialize the quantile estimator model with tail approximation
        """
        self.name = f"alpha={alpha}" + ("" if tail_approx is None else "_tail_approx") + ("" if not grad else "_grad")
        self.alpha = alpha
        self.gamma = gamma
        self.tail_approx = tail_approx
        self.plot_data = []
        self.fit_prob = None
        self.grad = grad


    def poly_fit(self, X, Y, G):
        # fit a quadratic based on the sampled points X and function values
        # Y and gradients G

        m = X.shape[0]

        if self.fit_prob is None:
            self.a = cp.Variable(nonneg=True, name="a")
            self.b = cp.Variable(name="b")
            self.c = cp.Variable(name="c")
            self.Xpsquare = cp.Parameter(shape=(m,), name="Xsquare")
            self.Xp = cp.Parameter(shape=(m,), name="X")
            self.Yp = cp.Parameter(shape=(m,), name="Y")
            self.Gp = cp.Parameter(shape=(m,), name="G")
        
            y_hat = cp.multiply(self.a, self.Xpsquare) + cp.multiply(self.b, self.Xp) + self.c
            g_hat = cp.multiply(2 * self.a, self.Xp) + self.b

            loss = cp.sum_squares(self.Yp - y_hat) + cp.sum_squares(self.Gp - g_hat)
            self.fit_prob = cp.Problem(cp.Minimize(loss))

        self.Xpsquare.value = X ** 2
        self.Xp.value = X
        self.Yp.value = Y
        self.Gp.value = G

        self.fit_prob.solve(solver=cp.MOSEK)

        # z = np.linspace(X.min(), X.max(), 100)
        # from matplotlib import pyplot as plt
        # plt.scatter(X, Y)
        # plt.plot(z, self.a.value * z ** 2 + self.b.value * z + self.c.value, label="fit_my")
        # plt.legend()
        # plt.show()

        return float(self.a.value), float(self.b.value), float(self.c.value)


    def fit(self, y, prev, tail_approx, prob = None):
        """
        Fit the model to the data y and return the prediction and the problem
        object for fast retraining
        """
        H_total = len(y)
        weight = self.gamma ** np.arange(H_total)[::-1]
        weight = weight / weight.sum()

        H = len(y) if tail_approx is None else tail_approx

        theta = cp.Variable(name="theta")
        pinball_losses = cp.maximum(0, theta - y[-H:]) * (1 - self.alpha) + cp.maximum(0, y[-H:] - theta) * self.alpha
        loss = cp.sum(cp.multiply(weight[-H:], pinball_losses))

        tail = 0.0
        tail_active = False
        if tail_approx is not None and len(y) > H:
            k = 3
            tail_active = True
            tail_losses = cp.maximum(0, theta - y[-k*H:-H]) * (1 - self.alpha) + cp.maximum(0, y[-k*H:-H] - theta) * self.alpha
            tail_loss = cp.sum(cp.multiply(weight[-k*H:-H], tail_losses))

            def tail_loss_fn(x):
                m = x.shape[0]
                y_tail = y[-k*H:-H][None, :]
                x_tail = x[:, None]
                weight_tail = weight[-k*H:-H]

                tail_losses_np = np.maximum(0, x_tail - y_tail) * (1 - self.alpha) + np.maximum(0, y_tail - x_tail) * self.alpha
                out = tail_losses_np @ weight_tail
                return out
            
            def tail_loss_fn_grad(x):
                """Compute the gradient of the tail loss function at x_tail"""
                m = x.shape[0]
                y_tail = y[-k*H:-H][None, :]
                x_tail = x[:, None]
                weight_tail = weight[-k*H:-H]

                d = weight_tail.shape[0]

                grad = np.zeros((m,d))
                xgy = x_tail > y_tail
                ygx = y_tail > x_tail

                grad[xgy] = 1 - self.alpha
                grad[ygx] = -self.alpha

                return grad @ weight_tail

            def f(x):
                theta.value = x
                return tail_loss.value
            
            # sample m random points from a normal distribution around prev
            m = 10
            np.random.seed(0)
            X = sp.stats.norm.rvs(prev, np.abs(prev)/5, size=m)
            Y = tail_loss_fn(X)
            G = tail_loss_fn_grad(X)

            if self.grad:
                a, b, c = self.poly_fit(X, Y, G)
            else:
                a,b,c = np.polyfit(X, Y, 2)
                a = np.maximum(a, 1e-6)

            if H_total in [100,200,300,400,500,600,700,800,900,998]:
                z = np.linspace(X.min(), X.max(), 100)
                XX = np.linspace(X.min(), X.max(), 100)
                YY = tail_loss_fn(XX)
                self.plot_data.append((X,Y, z, a * z ** 2 + b * z + c, prev, f(prev), XX,YY))

                # from matplotlib import pyplot as plt
                # plt.scatter(X, Y)
                # plt.plot(z, a * z ** 2 + b * z + c, label="fit_my")
                # plt.plot(z, a * z ** 2 + b * z + c, label="fit")
                # plt.legend()
                # plt.show()

            if prob is None:
                A = cp.Parameter(nonneg=True, name="A")
                B = cp.Parameter(name="B")
                C = cp.Parameter(name="C")

                Weight = cp.Parameter(shape=(H,), name="Weight", nonneg=True)
                Y_param = cp.Parameter(shape=(H,), name="Y_param")
                pinball_losses = cp.maximum(0, theta - Y_param) * (1 - self.alpha) + cp.maximum(0, Y_param - theta) * self.alpha
                loss = cp.sum(cp.multiply(Weight, pinball_losses))
                tail = A * cp.square(theta) + B * theta + C

                t = cp.Variable(pinball_losses.size ,name="t")
                loss = Weight @ t
                constraints = [t >= pinball_losses]
                objective = loss + tail
                prob = cp.Problem(cp.Minimize(objective), constraints)

            d = prob.param_dict
            d["A"].value = a
            d["B"].value = b
            d["C"].value = c
            d["Weight"].value = weight[-H:]
            d["Y_param"].value = y[-H:]

        else:
            objective = loss + tail
            prob = cp.Problem(cp.Minimize(objective))
            
        prob.solve(ignore_dpp=False, solver=cp.ECOS)
        theta_val = prob.var_dict["theta"].value
        if not tail_active:
            prob = None

        return theta_val, prob


    def _online_pred(self, y, tail_approx=None):
        """
        Fit the model to the data y and return the predictions and times
        """
        preds = np.zeros(y.shape[0])
        times = np.zeros(y.shape[0])
        prev = None
        prob = None
        print(self.name)
        for i in tqdm(range(1,y.shape[0])):
            y_train = y[:i]
            ts = time.time()
            pred, prob = self.fit(y_train, prev, tail_approx, prob)
            te = time.time()
            preds[i] = pred
            times[i] = te - ts
            prev = pred
        return preds, times

    def online_pred(self, X, y):
        """
        Wrapper for the online prediction method ignoring the X input
        """
        return self._online_pred(y, self.tail_approx)
        
            
class SparseInverseCov(Model):
    def __init__(self, lam, beta, name='SparseInverseCov'):
        """
        Initialize the sparse inverse covariance model
        """
        self.name = name
        self.lam = lam
        self.beta = beta
        self.prob = None

    def alpha_t(self, t):
        """
        Helper function to compute the alpha_t value
        """
        return (1-self.beta) /  (1-self.beta ** (t+1))


    def fit(self, XXt_ewma):
        """
        Fit the model to the compressed data XXt_ewma and return the predictions and times
        """
        if self.prob is None:
            # create a cvxpy problem to solve the sparse inverse covariance problem
            d = XXt_ewma.shape[0]
            theta = cp.Variable((d,d), symmetric=True)
            # create the off-diagonal regularizer
            off_diag = np.ones((d,d)) - np.eye(d)
            l1 = cp.norm(cp.multiply(theta, off_diag), 1)
            ld = cp.log_det(theta)

            X = cp.Parameter((d,d), name="X")
            loss = cp.trace(X @ theta)

            objective = loss - ld + self.lam * l1
            prob = cp.Problem(cp.Minimize(objective))

        X = prob.param_dict["X"]
        X.value = XXt_ewma
        prob.solve(solver=cp.MOSEK)

        return theta.value
    
    def online_pred(self, X, y):
        """
        Wrapper for the online prediction method ignoring the X input
        """
        return self._online_pred(y)

    def _online_pred(self, X):
        """
        Fit the model to the data X and return the predictions and times
        """
        d = X.shape[1]
        preds = np.zeros((X.shape[0],d,d))
        times = np.zeros(X.shape[0])
        XXt_ewma = np.outer(X[0], X[0])

        # start at index s so there is enough data to estimate a covariance
        # matrix of dimension d. There are d^2/2 parameters, so we need at
        s = d 
        for t in tqdm(range(s,X.shape[0])):
            w_1 = self.beta * (1-(self.beta ** (t)))/(1-(self.beta ** (t+1)))
            w_2 = (1 - self.beta) / (1 - (self.beta ** (t+1)))
            XXt_ewma = w_1 * XXt_ewma + w_2 * np.outer(X[t], X[t])

            ts = time.time()
            pred = self.fit(XXt_ewma)
            te = time.time()
            preds[t] = pred
            times[t] = te - ts
        
        return preds[s:], times[s:]

class SparseInverseCovDumb(Model):
    """
    Dumb version of the sparse inverse covariance model that doesn't use the
    recursive formulation
    """
    def __init__(self, lam, beta, name='SparseInverseCov'):
        self.name = name
        self.lam = lam
        self.beta = beta

    def alpha_t(self, t):
        """
        Helper function to compute the alpha_t value
        """
        return (1-self.beta) /  (1-self.beta ** (t+1))


    def fit(self, X):
        """
        Fit the model to the data X and return the predictions and solve time so
        as to not count compilation, which is inneficient
        """
        # create a cvxpy problem to solve the sparse inverse covariance problem
        d = X.shape[1]
        theta = cp.Variable((d,d), symmetric=True)
        # create the off-diagonal regularizer
        off_diag = np.ones((d,d)) - np.eye(d)
        l1 = cp.norm(cp.multiply(theta, off_diag), 1)
        ld = cp.log_det(theta)

        t = X.shape[0]
        loss = cp.sum(
            [(self.beta ** (t-tau)) * (cp.trace(np.outer(X[tau-1],X[tau-1]) @ theta) - ld) for tau in range(1,t+1)]
        )
        a_t = self.alpha_t(t)
        objective = a_t* loss + self.lam * l1
        prob = cp.Problem(cp.Minimize(objective))

        prob.solve(solver=cp.MOSEK)

        return theta.value, prob.solver_stats.solve_time
    
    def online_pred(self, X, y):
        """
        Wrapper for the online prediction method ignoring the X input
        """
        return self._online_pred(y)

    def _online_pred(self, X):
        """
        Fit the model to the data X and return the predictions and times
        """
        d = X.shape[1]
        preds = np.zeros((X.shape[0],d,d))
        times = np.zeros(X.shape[0])

        # start at index s so there is enough data to estimate a covariance
        # matrix of dimension d
        s = d 
        for t in tqdm(range(s,X.shape[0])):
            pred,tm  = self.fit(X[:t])
            preds[t] = pred
            times[t] = tm
        
        return preds[s:], times[s:]
