"""
_opt_log.py

This module contains the optimisation-specific functions for logistic.py and logistic_fw.py

Functions:
- sigmoid(t): Returns the sigmoid mapping between t and w.
- logit(w): Returns the logit mapping between w and t.
- h(u, delta_frac): A helper function for the refined gradient expression.
- f_grad_cg(): Computes the gradient expression, upsilon and numpy arrays to support the Woodbury matrix identity.
- adam(): Employs gradient descent with the Adam optimisers to minimise the novel objective function.
- dynamic_grid(): Conducts a search over a dynamic grid of lambda values.
- bss(): Manages the dynamic grid search, and conducts model evaluation.

"""
import numpy as np
from numpy.linalg import pinv, norm
import time
from scipy.sparse.linalg import cg, LinearOperator
from scipy.optimize import LinearConstraint, minimize
from sklearn.linear_model import LogisticRegression


class Result:
    def __init__(self, models, niters):
        self.models = models
        self.niters = niters

def logit(v):
	""" 
	Logit function applied elementwise 
	"""

	w = np.log(v/(1-v))
	return w


def sigmoid(w):
	""" 
	Sigmoid function applied elementwise 
	"""
    
	v = 1/(1+np.exp(-w))
	return v

def log_lh(X, y, beta):
    """
    Compute log-likelihood for logistic regression (Bernoulli).
    
    Args:
        X (np.ndarray): Design matrix (n_samples, n_features).
        y (np.ndarray): Binary labels (0 or 1) (n_samples,).
        theta (np.ndarray): Model parameters (n_features,).
    
    Returns:
        float: Log-likelihood.
    """
    z = X @ beta  # Xθ
    p = sigmoid(z)  # Sigmoid
    
    log_loss = np.sum(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))  # Avoid log(0)
    
    return log_loss

def ht(t, X, y, beta, delta):
    '''
    Function h_delta(t, beta)
    '''
    
    t_ext = np.concatenate(([1], t))
    tbeta = t_ext*beta
    penalty = delta*np.dot((1 - t_ext*t_ext)*beta, beta)
    
    h = -log_lh(X, y, tbeta)
    h = h + penalty
    return h

def grad_ht(t, X, y, beta, delta):
    '''
    Gradient of ht
    '''
    t_ext = np.concatenate(([1], t))
    tbeta = t_ext*beta
    lin_rel = X@tbeta
    probs = sigmoid(lin_rel)
    
    grad =  X.T@(y - probs)
    grad = - t_ext*grad
    grad = grad + 2*delta*(1 - t_ext*t_ext)*beta

    return grad


#%%
'''

This cell has 3 versions of optimization methods for finding beta_tilde for a given t
    1.  beta_tilde_gd is simple basic gradient descent
    2.  beta_tilde_adam uses ADAM optimizer
    3.  beta_tilde_sp is more advanced using L-BFGS via minimize command of SciPy package
    4.  beta_tilde_sklearn uses sklearn's L-BFGS solver
    
Conclusion: From simulations, we observed that SciPy based optimization is much better in the sense 
            that the output is not sensitive to the initial value. 
    
'''

def beta_tilde_gd(t, X, y, beta_start, delta, 
                  
                  patience = 10,
                  # Basic gradient descent parameters
                  alpha = 0.1, 
                  gd_tol = 0.00001, 
                  gd_maxiter = 1000000):
    '''
    Grdaient descent for obtaining beta_tilde_t at a given t
    '''
    beta_init = beta_start.copy()
    beta_prev = beta_init.copy()
    count = 0
    for _ in range(gd_maxiter):
        grad_h = grad_ht(t, X, y, beta_prev, delta)
        beta = beta_prev - alpha*grad_h

        if np.linalg.norm(beta - beta_prev) < gd_tol:
            count += 1
        beta_prev = beta.copy()
        
        if count >= patience:
            
            return beta, True
    
    if count < patience:
        converge = False
    else:
        converge = True
    
    return beta, converge
    
def beta_tilde_adam(t, X, y, beta_start, delta, 
    
                    patience = 10,
                    
                    # ADAM parameters
                    alpha = 0.1, 
                    xi1 = 0.9, 
                    xi2 = 0.999,            
                    epsilon = 10e-8,
                    adam_tol = 0.001, 
                    adam_maxiter = 1000):
    '''
    ADAM optimization for obtaining beta_tilde_t at a given t
    '''
    
    # Initializing
    beta_init = beta_start.copy()
    beta_prev = beta_init.copy()
    count = 0
    d = beta_init.shape[0]
    u = np.zeros(d)
    v = np.zeros(d)
    beta = np.zeros(d)
    
    for l in range(adam_maxiter):
        grad_h = grad_ht(t, X, y, beta_prev, delta)
        
        ## Momentum and momentum square updates
        u = xi1*u - (1 - xi1)*grad_h
        v = xi2*v + (1 - xi2)*(grad_h*grad_h) 
    
        u_hat = u/(1 - xi1**(l+1))
        v_hat = v/(1 - xi2**(l+1))
        
        
        beta = beta_prev + alpha*np.divide(u_hat, epsilon + np.sqrt(v_hat))

        if np.linalg.norm(beta - beta_prev) < adam_tol:
            count += 1
        beta_prev = beta.copy()
        
        if count >= patience:
            return beta, True
        
    if count < patience:
        converge = False
    else:
        converge = True

    return beta, converge

def beta_tilde_sp(t, X, y, beta_start, delta,
                   
                   # Minimize parameters
                   method='L-BFGS-B',
                   sp_tol=0.0001):
    '''
    Function for obtaining the beta_tilde_t using L-BFGS-B via SciPy package
    '''
    
    grad_h = lambda v: grad_ht(t, X, y, v, delta)
    objective = lambda v: ht(t, X, y, v, delta)
    
    beta_init = beta_start.copy()
    #p = beta_init.shape[0]
    #hessp_h = LinearOperator((p, p), matvec= lambda v: hessp_h_t(v, beta, t, X, y, delta))
    result = minimize(objective, beta_init, jac=grad_h, method=method, tol=sp_tol)
    
    return result.x, True


def beta_tilde_sklearn(t,X, y, delta,
                            *, clip_eps=1e-6,
                            max_iter=100000, tol=1e-16, **extra):
    """
    Fit the penalised logistic objective with sklearn's solver.

    Parameters
    ----------
    X : (n_samples, n_features) – NO intercept column
    y : (n_samples,)           – 0/1 or ±1 labels
    t : (n_features,)          – 0 < t_j < 1
    delta : float > 0          – global multiplier δ
    clip_eps : float           – guard for t_j close to 0 or 1
    max_iter, tol, **extra     – forwarded to LogisticRegression

    Returns
    -------
    betatilde_hat  : (p+1,)   (intercept first)
    converged : bool    – True if the solver stopped before `max_iter`
    """
    # 0. validation ----------------------------------------------------
    t = np.asarray(t, float).ravel()
    if t.size != X.shape[1]:
        raise ValueError("`t` length must match n_features")
    if np.any((t <= 0) | (t >= 1)):
        raise ValueError("all t_j must lie in (0,1)")
    if delta <= 0:
        raise ValueError("`delta` must be positive")
		
    scale_vec = np.sqrt((1 - t**2) / t**2)
    X_scaled = X.astype(float)
    if hasattr(X_scaled, "multiply"):          # sparse matrix
        X_scaled = X_scaled.multiply(1.0 / scale_vec)
    else:                                      # dense ndarray
        X_scaled /= scale_vec

    
    # 2. C that matches δ ---------------------------------------------
    C_match = 1.0 / (2.0 * delta)

    clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=C_match,
        fit_intercept=True,
        max_iter=max_iter,
        tol=tol,
        **extra
    ).fit(X_scaled, y)

    # 3. back-transform to original scale ------------------------------
    gamma_hat = clf.coef_.ravel() / scale_vec
    betatilde_hat = gamma_hat/t
    b_hat = float(clf.intercept_[0])
    betatilde_hat = np.concatenate(([b_hat], betatilde_hat))

    # 4. convergence flag ---------------------------------------------
    # For binary class, clf.n_iter_ is length-1 array
    converged = clf.n_iter_[0] < max_iter

    return betatilde_hat, converged

#%%
"""
Now focus on computing the gradient
"""
def grad_llh(beta, X, y):
    '''
    Gradient of ht
    '''
    z = X@beta
    probs = sigmoid(z)
    grad =  X.T@(y - probs)

    return grad 


def hessian_gt_gamma_vec_prod(v, t,  X, gamma, delta):
    
    t_ext = np.concatenate(([1], t))
    Dtv = (1/(t_ext*t_ext) - 1)*v
    
    Xv = X @ v
    z = X @ gamma
    probs = sigmoid(z)
    
    sig_inv = (probs*(1 - probs))
    result = X.T @ (sig_inv*Xv) + 2*delta*Dtv
    
    return result
    

def hessian_gt_gamma_inv_vec_prod(v, t, X, gamma, delta,
                                  
                                  # Parameters for cg
                                  cg_maxiter=None, 
                                  cg_tol=1e-5):
    
    """
    Function for computing g"(gamma_t)^-1 v
    """
    size = gamma.shape[0]
    if cg_maxiter == None:
        cg_maxiter = size
    
    matvec = lambda v: hessian_gt_gamma_vec_prod(v, t, X, gamma, delta)
    A = LinearOperator((size, size), matvec=matvec)

    u, _ = cg(A, v, maxiter=cg_maxiter, rtol=cg_tol)
        
    return u

def f_grad_cg(t, beta_start, X, y, delta, 
                solver = "adam", 
                # Parameters for conjugate gradient
                cg_maxiter=None,
                cg_tol=1e-5):
    
    """
    This function computes the gradient of f_lam at a given t
    """
    

    n = y.shape[0]
    
    if solver == "lbfgs":
        beta_tilde, _ = beta_tilde_sp(t, X, y, beta_start, delta)
    elif solver == "adam":
        beta_tilde, _ = beta_tilde_adam(t, X, y, beta_start, delta)
    elif solver == "sklearn":
        beta_tilde, _ = beta_tilde_sklearn(t, X[:, 1:], y,  delta)
    else: 
        print("solver option for beta estimation is not valid. Using 'adam'")
        beta_tilde, _ = beta_tilde_adam(t, X, y, beta_start, delta)

    
    t_ext = np.concatenate(([1], t))
    gamma = t_ext*beta_tilde # gamma_t in the paper. Step 1 
    vec = grad_llh(gamma, X, y) # step 2 
    v = hessian_gt_gamma_inv_vec_prod(vec, t, X, gamma, delta, cg_maxiter=cg_maxiter, cg_tol=cg_tol) # step 3
    #print(f"grad_l:     = {vec[0:10]}")
    t3 = t*t*t
    f_grad = (-4*delta/n)*(v[1:]*gamma[1:]/t3)
    
    return f_grad, beta_tilde



def f(t, X, y, delta, solver = "adam"):
    
    (n, p) = X.shape
    beta_start = np.zeros(p)
    
    if solver == "lbfgs":
        beta_tilde, _ = beta_tilde_sp(t, X, y, beta_start, delta)
    elif solver == "adam":
        beta_tilde, _ = beta_tilde_adam(t, X, y, beta_start, delta)
    elif solver == "sklearn":
        beta_tilde, _ = beta_tilde_sklearn(t, X[:, 1:], y,  delta)
    else: 
        print("solver option for beta estimation is not valid. Using 'adam'")
        beta_tilde, _ = beta_tilde_adam(t, X, y, beta_start, delta)
        
    t_ext = np.concatenate(([1], t))
    gamma = t_ext*beta_tilde
    
    val = (-1/n)*log_lh(X, y, gamma)
    
    return val


#%%
def fw(X, y, q, 
        max_iter = 1000, 
        r=1.5, 
        m=10, 
        scale = True,
        delta_min = 0.02,    # smallest delta value on the grid
        alpha=0.05,           # line search is applied if true 
        patience = 10,        # patience parameter to terminate the algorithm
        solver = 'adam',      # solver for obtaining beta for each t
        cg_maxiter=None,     # Maximum number of iterations allowed by CG
        cg_tol=0.001,
        verbose = True):        # Tolerance of CG

    (n, p) = X.shape
    
    # Setting default values
    if cg_maxiter == None:
        cg_maxiter = p
        

    if scale:
        column_norms = np.linalg.norm(X[:, 1:], axis=0) 
        X_norm = X.copy() # skip first column
        X_norm[:, 1:] = X_norm[:, 1:] / column_norms
        
    model_list = []
    niter_list = []
    #print(X_norm[0,:])
    for k in range(1, q+1):
        if verbose:
            print("Model size k = ", k)
        t = np.ones(p-1)*(k/(p-1))

            
        beta_start = np.zeros(p)
        delta = delta_min
        
        l = 0
        count = 0
        stop = False
        while not stop:
            
            if l%m == 0:
                delta = delta*r
            grad, beta_start = f_grad_cg(t, beta_start, X_norm, y, delta, 
                                         solver = solver, cg_maxiter=cg_maxiter, cg_tol=cg_tol)


            model = np.argsort(grad)[:k]  # Indices of k smallest elements of the gradient
            s = np.zeros(p-1, dtype=int)
            s[model] = 1
            
            # Update t
            t = (1 - alpha) * t + alpha * s


            t[t < 0.0001] = 0.0001 # truncation for numerical stability
            t[t > 0.9999] = 0.9999 # truncation for numerical stability
                
            l += 1
            if np.any((t >= 0.001) & (t <= 0.999)):
                count = 0
            else:
                count += 1
            
            if l >= max_iter or count >= patience:
                stop = True
        if verbose:
            print("Model:", np.sort(model)+1)
        model_list.append(np.sort(model)+1)
        niter_list.append(l)
    
    result = Result(model_list, niter_list)
    return  result






#%%

def adam(X, y,  lam, t_init,
        delta_frac = 1,

        ## Adam parameters
        xi1 = 0.9, 
        xi2 = 0.999,            
        alpha = 0.1, 
        epsilon = 10e-8,
     
        ## Parameters for Termination
        gd_maxiter = 100000,
        gd_tol = 1e-5,
        max_norm = True,    # By default, we use max norm as the termination condition.
        patience=10,
        
        ## Truncation parameters
        tau = 0.5,
        eta = 0.0, 
        
        ## Parameters for Conjugate Gradient method
        cg_maxiter = None,
        cg_tol = 1e-5):
    
    """ The Adam optimiser used within the COMBSS algorithm.

    Parameters
    ----------
    X : array-like of shape (n, p)
        The design matrix, where `n` is the number of samples in the dataset
        and `p` is the number of covariates measured in each sample.
    y : array-like of shape (n, )
        The response data, where `n` is the number of samples in the dataset.

    lam : float
        The penalty parameter used within the objective function. Referred to as
        'lambda' in the paper Moka et al. (2024).

    t_init : array-like of floats of shape (p, )
        The initial values of t passed into Adam.
        Default value = [].

    delta_frac : float
        The value of n/delta as found in the objective function in the COMBSS algorithm.
        Default value = 1.
    
    xi1 (Adam parameter) : float
        The exponential decay rate for the first moment estimates in Adam. 
        Default value = 0.9.

    xi2 (Adam parameter) : float
        The exponential decay rate for the second-moment estimates.
        Default value = 0.99.

    alpha (Adam parameter) : float
        The learning rate for Adam.
        Default value = 0.1.

    epsilon (Adam parameter) : float
        A small number used to avoid numerical instability when dividing by 
        extremely small numbers within Adam.
        Default value = 1e-8.

    gd_maxiter : int
        The maximum number of iterations for Adam before the algorithm terminates.
        Default value = 100000.

    gd_tol : float
        The acceptable tolerance used for the termination condition in Adam.
        Default value = 1e-5.

    max_norm : Boolean
        Boolean value that signifies if the max norm is used for the termination condition in gradient descent.
        If max_norm = True, the termination condition is evaluated using the max norm. Otherwise, 
        the L2 norm will be used instead.
        Default value = True

    patience : int
        The integer that specifies how many consecutive times the termination condiiton has to be satisfied
        before the function terminates.
        Default value = 10.

    tau : float
        The cutoff value for t that signifies its selection of the covariates. 
        If t[i] > tau, the ith covariate is selected. 
        If t[i] < tau, the ith covariate is not selected.
        Default value = 0.5.

    eta : float
        The parameter that dictates the upper limit used for truncating matrices.
        If the value of t[i] is less than eta, t[i] will be approximated to zero,
        and the ith column of X will be ignored in calculations to improve algorithm perfomance.
        Default value = 0.

    cg_maxiter : int
        The maximum number of iterations for the conjugate gradient algortihm.
        Default value = n.

    cg_tol : float
        The acceptable tolerance used for the termination condition in the conjugate gradient 
        algortihms.
        Default value = 1e-5.


    Returns
    -------
    t : array-like of shape (p, )
        The array of t values at the conclusion of the Adam optimisation algorithm.

    subset : array-like of integers
        The final chosen subset, in the form of an array of integers that correspond to the 
        indicies chosen after using the Adam optimiser.

    converge : Boolean 
        Boolean value that signifies if the gradient descent algorithm terminated by convergence 
        (converge = True), or if it exhausted its maximum iterations (converge = False).

    l+1 : int
        The number of gradient descent iterations executed by the algorithm. If the algorithm 
        reaches the maximum number of iterations provided into the function, l = gd_maxiter.
    """
    
    (n, p) = X.shape
    if cg_maxiter == None:
        cg_maxiter = n
    
    ## One time operation
    Xy = (X.T@y)/n

    
    ## Initialization
    t = t_init.copy()
        
    w = sigmoid(t)
    
    t_trun = t.copy()
    t_prev = t.copy()
    active = p
    
    u = np.zeros(p)
    v = np.zeros(p)
    
    gamma_trun = np.zeros(p)  

    upsilon = np.zeros(p)
    g1 = np.zeros(n)
    g2 = np.zeros(n)
    
    delta = delta_frac*n
    
    count_to_term = 0
    
    
    for l in range(gd_maxiter):
        M = np.nonzero(t)[0] # Indices of t that correspond to elements greater than eta. 
        M_trun = np.nonzero(t_trun)[0] 
        active_new = M_trun.shape[0]
        
        if active_new != active:
            ## Find the effective set by removing the columns and rows corresponds to zero t's
            X = X[:, M_trun]
            Xy = Xy[M_trun]
            active = active_new
            t_trun = t_trun[M_trun]
        
        ## Compute gradient for the effective terms
        grad_trun, gamma_trun, upsilon, g1, g2 = f_grad_cg(t_trun, X, y, Xy, delta_frac, gamma_trun[M_trun],  upsilon[M_trun], g1=g1, g2=g2, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
        grad_trun = grad_trun + lam
        w_trun = w[M]
        grad_trun = grad_trun*(logit(w_trun)*(1 - logit(w_trun)))
        
        ## ADAM Updates 
        u = xi1*u[M_trun] + (1 - xi1)*grad_trun
        v = xi2*v[M_trun] + (1 - xi2)*(grad_trun*grad_trun) 
    
        u_hat = u/(1 - xi1**(l+1))
        v_hat = v/(1 - xi2**(l+1))
        
        w_trun = w_trun - alpha*np.divide(u_hat, epsilon + np.sqrt(v_hat)) 
        w[M] = w_trun
        t[M] = logit(w_trun)
        
        w[t <= eta] = -np.inf
        t[t <= eta] = 0.0

        t_trun = t[M] 
        
        if max_norm:
            norm_t = max(np.abs(t - t_prev))
            if norm_t <= gd_tol:
                count_to_term += 1
                if count_to_term >= patience:
                    break
            else:
                count_to_term = 0
                
        else:
            norm_t = norm(t)
            if norm_t == 0:
                break
            
            elif norm(t_prev - t)/norm_t <= gd_tol:
                count_to_term += 1
                if count_to_term >= patience:
                    break
            else:
                count_to_term = 0
        t_prev = t.copy()
    
    subset = np.where(t > tau)[0]

    if l+1 < gd_maxiter:
        converge = True
    else:
        converge = False
    return  t, subset, converge, l+1


def dynamic_grid(X, y, t_init,
                   q = None,
                   nlam = None,
                   tau=0.5,             # tau parameter
                   delta_frac=1,        # delta_frac = n/delta
                   fstage_frac = 0.5,   # Fraction lambda values explored in first stage of dynamic grid
                   eta=0.0,             # Truncation parameter
                   patience=10,         # Patience period for termination 
                   gd_maxiter=1000,     # Maximum number of iterations allowed by GD
                   gd_tol=1e-5,         # Tolerance of GD
                   cg_maxiter=None,     # Maximum number of iterations allowed by CG
                   cg_tol=1e-5):        # Tolerance of CG
    
    """ Executes the COMBSS algorithm over a dynamic grid of lambdas to provide a subset for each lambda on the grid.

    The dynamic grid of lambda is generated as follows: We are given maximum subset size q of interest. 
    
    First pass: We start with $\lambda = \lambda_{\max} = \mathbf{y}^\top \mathbf{y}/n$, 
                where an empty subset is guaranteed to be selected, and decrease iteratively $\lambda \leftarrow \lambda/2$ 
                until we find subset of size larger than $q$. 
    
    Second pass: Then, suppose $\lambda_{grid}$ is (sorted) vector of $\lambda$ valued exploited in 
                 the first pass, we move from the smallest value to the large value on this grid, 
                 and run COMBSS at $\lambda = (\lambda_{grid}[k] + \lambda_{grid}[k+1])/2$ if $\lambda_{grid}[k]$ 
                 and $\lambda_{grid}[k+1]$ produced subsets with different sizes. 
                 We repeat this until the size of $\lambda_{grid}$ is larger than a fixed number $nlam$.

    Parameters
    ----------
    X : array-like of shape (n, p)
        The design matrix, where `n` is the number of samples in the dataset
        and `p` is the number of covariates measured in each sample.
    y : array-like of shape (n, )
        The response data, where `n` is the number of samples in the dataset.
        
    q : int
        The maximum subset size of interest. If q is not provided, it is taken to be n.
        Default value = min(n, p).

    nlam : float
        The number of lambdas explored in the dynamic grid. 
        Default value = None.

    t_init : array-like of floats of shape (p, )
        The initial values of t passed into Adam.
        Default value = 0.5*np.ones(p).

    tau : float
        The cutoff value for t that signifies its selection of covariates. 
        If t[i] > tau, the ith covariate is selected. 
        If t[i] < tau, the ith covariate is not selected.
        Default value = 0.5.

    delta_frac : float
         The value of n/delta as found in the objective function for COMBSS.
        Default value = 1.

    fstage_frac : float
        The fraction of lambda values explored in first pass of dynamic grid.
        Default value = 0.5.

    eta : float
        The parameter that dictates the upper limit used for truncating matrices.
        If the value of t[i] is less than eta, t[i] will be approximated to zero,
        and the ith column of X will be ignored in calculations to improve algorithm perfomance.
        Default value = 0.

    patience : int
        The integer that specifies how many consecutive times the termination conditon on the norm has 
        to be satisfied before the function terminates.
        Default value = 10.

    gd_maxiter : int
        The maximum number of iterations for Adam before the algorithm terminates.
        Default value = 1000.

    gd_tol : float
        The acceptable tolerance used for the termination condition in Adam.
        Default value = 1e-5.

    cg_maxiter : int
        The maximum number of iterations for the conjugate gradient algortihm.
        Default value = n.

    cg_tol : float
        The acceptable tolerance used for the termination condition in the conjugate gradient 
        algortihms.
        Default value = 1e-5.

        
    Returns
    -------
    subset_list : array-like of array-like of integers. 
        Describe the indices chosen as the subsets for each lambda, e.g. [[1], [1, 6], [1, 11, 20], [12]]  

    lam_list : array-like of floats.
        Captures the sequence of lambda values explored in best subset selection.
    """
    (n, p) = X.shape
    
    # If q is not given, take q = n.
    if q == None:
        q = min(n, p)
    
    # If number of lambda is not given, take it to be n.
    if nlam == None:
        nlam == 50

    # If t_init is not given, we take t_init to be the p-dimensional vector with 0.5 as every element.
    if t_init.shape[0] == 0:
        t_init = np.ones(p)*0.5
    
    if cg_maxiter == None:
        cg_maxiter = n
    
    # Maximal value for lambda
    lam_max = y@y/n 

    # Lists to store the findings
    subset_list = []
    
    lam_list = []
    lam_vs_size = []
    
    lam = lam_max
    count_lam = 0

    ## First pass on the dynamic lambda grid
    stop = False
    non_empty_set = False
    while not stop:

        t_final, subset, converge, _ = adam(X, y, lam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, patience=patience, gd_maxiter=gd_maxiter, gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)

        len_subset = subset.shape[0]
        if len_subset != 0:
            non_empty_set = True
        
        lam_list.append(lam)
        subset_list.append(subset)
        lam_vs_size.append(np.array((lam, len_subset)))
        count_lam += 1
        
        if len_subset >= q: 
            stop = True
        
        if count_lam > nlam*fstage_frac and non_empty_set:
            stop = True
        lam = lam/2
        
        
    stop = False
    if count_lam >= nlam or not non_empty_set:
        stop = True

    ## Second pass on the dynamic lambda grid if stop = False
    while not stop:

        temp = np.array(lam_vs_size)
        order = np.argsort(temp[:, 0])
        lam_vs_size_ordered = temp[order]        

        ## Find the next index
        for i in range(order.shape[0]-1):

            if count_lam <= nlam and lam_vs_size_ordered[i+1][1] <= q and  (lam_vs_size_ordered[i+1][1] != lam_vs_size_ordered[i][1]):

                lam = (lam_vs_size_ordered[i][0] + lam_vs_size_ordered[i+1][0])/2

                t_final, subset, converge, _ = adam(X, y, lam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, patience=patience, gd_maxiter=gd_maxiter,gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)

                len_subset = subset.shape[0]

                lam_list.append(lam)
                subset_list.append(subset)
                lam_vs_size.append(np.array((lam, len_subset)))    
                count_lam += 1

            if count_lam > nlam:
                stop = True
                break

    temp = np.array(lam_vs_size)
    order = np.argsort(temp[:, 1])
    subset_list = [subset_list[i] for i in order]
    lam_list = [lam_list[i] for i in order]
    
    return  (subset_list, lam_list)


def bss(X_train, y_train, X_test, y_test, 
            q = None,           # Maximum subset size
            nlam = 50,          # Number of values in the lambda grid
            t_init= [],         # Initial t vector
            scaling = False,    # If True, the training data is scaled 
            tau=0.5,            # Threshold parameter
            delta_frac=1,       # delta_frac = n/delta
            eta=0.001,          # Truncation parameter
            patience=10,        # Patience period for termination 
            gd_maxiter=1000,    # Maximum number of iterations allowed by GD
            gd_tol=1e-5,        # Tolerance of GD
            cg_maxiter=None,    # Maximum number of iterations allowed by CG
            cg_tol=1e-5):
    
    """ Best subset selection from the list of subsets generated by COMBSS with 
        SubsetMapV1 as proposed in the original paper Moka et al. (2024)
        over a grid of dynamically generated lambdas.

        
    Parameters
    ----------
    X_train : array-like of shape (n_train, p)
        The design matrix used for training, where `n_train` is the number of samples 
        in the training data and `p` is the number of covariates measured in each sample.

    y_train : array-like of shape (n_train)
        The response data used for training, where `n_train` is the number of samples in the training data.

    X_test : array-like of shape (n_test, p)
        The design matrix used for testing, where `n_test` is the number of samples 
        in the testing data and `p` is the number of covariates measured in each sample.

    y_test : array-like of shape (n_test)
        The response data used for testing, where `n_test` is the number of samples in the testing data.    

    q : int
        The maximum subset size of interest. If q is not provided, it is taken to be n.
        Default value = min(n_train, p).

    nlam : int
        The number of lambdas explored in the dynamic grid.
        Default value = 50.

    t_init : array-like of integers
        The initial value of t passed into Adam optimizer.
        Default value = 0.5*np.ones(p).

    tau : float
        The cutoff value for t that signifies its selection of covariates. 
        If t[i] > tau, the ith covariate is selected. 
        If t[i] < tau, the ith covariate is not selected.
        Default value = 0.5.

    delta_frac : float
        The value of n_train/delta as found in the objective function for COMBSS.
        Default value = 1.

    eta : float
        The parameter that dictates the upper limit used for truncating t elements during the optimization.
        If the value of t[i] is less than eta, t[i] will be mapped to zero,
        and the ith column of X will be removed to improve algorithm perfomance.
        Default value = 0.

    patience : int
        The integer that specifies how many consecutive times the termination condiiton has to be satisfied
        before the Adam optimzer terminates.
        Default value = 10.

    gd_maxiter : int
        The maximum number of iterations for Adam before the algorithm terminates.
        Default value = 1000.

    gd_tol : float
        The acceptable tolerance used for the termination condition in Adam.
        Default value = 1e-5.

    cg_maxiter : int
        The maximum number of iterations for the conjugate gradient algortihm.
        Default value = n_train.

    cg_tol : float
        The acceptable tolerance used for the termination condition in the conjugate gradient 
        algortihms.
        Default value = 1e-5.


    Returns 
    -------
    A dictionary consisting of the following:
        
    subset_opt : array-like of integers
        The indices of the optimal subset of predictors chosen from all the subsets selected 
        by COMBSS over the dynamic grid of lambdas, 

    mse_opt : float
        The mean squared error on the test data corresponds to the subset_opt.

    beta_opt : array-like of floats  
        Represents estimates of coefficients for linear regression for the subset_opt.

    lam_opt : float
        The value of lambda corresponds to the subset_opt.

    time : float
        The time taken to execute COMBSS on the dynamic grid.
        
    lambda_list : list 
        The list of lambda values of the dynamic grid.
        
    subset_list : list
        The list subsets obtained by COMBSS on the dynamic grid of lambda values. For each i, 
        subset_list[i] corresponds to lambda_list[i].
    """
    
    if scaling:
        column_norms = np.linalg.norm(X_train, axis=0)
        X_train = X_train / column_norms

    (n, p) = X_train.shape
    t_init = np.array(t_init) 
    
    if t_init.shape[0] == 0:
        t_init = np.ones(p)*0.5
        
    # If q is not given, take q = n
    if q == None:
        q = min(n, p)
    
    tic = time.process_time()
    (subset_list, lam_list) = dynamic_grid(X_train, y_train, q = q, nlam = nlam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, patience=patience, gd_maxiter= gd_maxiter, gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
    toc = time.process_time()

    if scaling:
        X_train = X_train * column_norms

    """
    Computing the MSE on the test data
    """
    nlam = len(lam_list)
    mse_list = [] 
    beta_list = []
    
    for i in range(nlam):
        subset_final = subset_list[i]

        X_hat = X_train[:, subset_final]
        X_hatT = X_hat.T

        X_hatTy = X_hatT@y_train
        XX_hat = X_hatT@X_hat
    
        beta_hat = pinv(XX_hat)@X_hatTy 
        X_hat = X_test[:, subset_final]
        mse = np.square(y_test - X_hat@beta_hat).mean()
        mse_list.append(mse)
        beta_pred = np.zeros(p)
        beta_pred[subset_final] = beta_hat
        beta_list.append(beta_pred)

    ind_opt = np.argmin(mse_list)
    lam_opt = lam_list[ind_opt]
    subset_opt = subset_list[ind_opt]
    mse_opt = mse_list[ind_opt] 
    beta_opt = beta_list[ind_opt]
    
    time_taken = toc - tic
    
    result = {
        "subset" : subset_opt,
        "mse" : mse_opt,
        "coef" : beta_opt,
        "lambda" : lam_opt,
        "time" : time_taken,
        "subset_list" : subset_list,
        "lambda_list" : lam_list
        }

    return result
