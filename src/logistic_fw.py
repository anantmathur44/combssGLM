"""
logistic_fw.py

This module contains a user-friendly interface for using the model class to conduct best 
subset selection via Frank-Wolfe continuous optimisation for logistic regression.

All optimisation logic is handled in the private _opt_log.py module.
"""

import _opt_log as olog
import time


class model:
    """
        Model is a class implementation that enables users to run Frank-Wolfe algorithm
        for best subset selection in Python. Model contains the following method:
        - fit
        
        and attributes
        - run_time
        - subset_list
        - niter_list
    """
    def __init__(self):
        """
        The class object that creates a model useful for fitting high-dimensional data to obtain a sparse subset of predictors,
        optimised via Frank-Wolfe algorithm.
        """
        self.run_time = None
        self.subset_list = None
        self.niter_list = None
        
    def fit(self, X_train, y_train,
                q = None,           # maximum subset size
                max_iter = 1000,    # maximum iterations for Frank-Wolfe
                r = 1.5,            # parameter for Frank-Wolfe
                m = 10,             # parameter for Frank-Wolfe
                scale = True,       # If True, the training data is scaled 
                delta_min = 0.02,   # smallest delta value on the grid
                alpha = 0.05,       # line search parameter
                patience = 10,      # Patience period for termination 
                solver = 'adam',    # solver for obtaining beta for each t
                cg_maxiter = None,  # Maximum number of iterations allowed by CG
                cg_tol = 0.001,     # Tolerance of CG
                verbose = True):    # verbose output
        
        """ 
        Fits the data to select a best subset of covariates using the Frank-Wolfe algorithm.
        The fit function calls the fw function from _opt_log.
            
        Parameters
        ----------
        X_train : array-like of shape (n_train, n_covariates)
            The design matrix used for training, where `n_train` is the number of samples 
            in the training data and `n_covariates` is the number of covariates measured in each sample.

        y_train : array-like of shape (n_train)
            The response data used for training, where `n_train` is the number of samples in the training data.

        X_test : array-like of shape (n_test, n_covariates)
            The design matrix used for testing, where `n_test` is the number of samples 
            in the testing data and `n_covariates` is the number of covariates measured in each sample.

        y_test : array-like of shape (n_test)
            The response data used for testing, where `n_samples` is the number of samples in the testing data.    

        q : int
            The maximum subset size of interest. If q is not provided, it is taken to be the lesser value between 
            n, the number of observations in X_train, and p, the number or predictors in X_train.
            Default value = min(n,p).

        max_iter : int
            The maximum number of iterations for the Frank-Wolfe algorithm.
            Default value = 1000.

        r : float
            Parameter for Frank-Wolfe algorithm that controls delta scaling.
            Default value = 1.5.

        m : int
            Parameter for Frank-Wolfe algorithm that controls delta update frequency.
            Default value = 10.

        scale : bool
            Determines whether or not feature scaling is applied for optimisation.
            Default value = True.

        delta_min : float
            The smallest delta value on the grid for Frank-Wolfe algorithm.
            Default value = 0.02.

        alpha : float
            Line search parameter for Frank-Wolfe algorithm.
            Default value = 0.05.

        patience : int
            The integer that specifies how many consecutive times the termination condition has to be satisfied
            before the algorithm terminates.
            Default value = 10.

        solver : str
            The solver for obtaining beta for each t ('adam', 'lbfgs', 'sklearn').
            Default value = 'adam'.

        cg_maxiter : int
            The maximum number of iterations for the conjugate gradient algorithm.
            Default value = None.

        cg_tol : float
            The acceptable tolerance used for the termination condition in the conjugate gradient 
            algorithms.
            Default value = 0.001.


        After fitting
        -------------
        Once fitting is performed as previously demonstrated, each attribute can be directly accessed by running 
        model.<attribute>. For example, the models list can be obtained by calling model.subset_list. 
        The list of attributes for the model class is as follows,

        subset_list : list of arrays
            The list of subsets obtained by Frank-Wolfe algorithm for each model size k=1,2,...,q.
            Each element contains the indices of selected features for that model size.

        niter_list : list of integers
            The list of iteration counts for each model size k=1,2,...,q.
            Each element contains the number of iterations needed for convergence for that model size.

        run_time : float
            The time taken to execute the Frank-Wolfe algorithm.

        """

        print("Fitting the model using Frank-Wolfe algorithm ...")
        start_time = time.time()
        result = olog.fw(X_train, y_train, q=q, max_iter=max_iter, r=r, m=m, scale=scale, 
                        delta_min=delta_min, alpha=alpha, patience=patience, solver=solver, 
                        cg_maxiter=cg_maxiter, cg_tol=cg_tol, verbose=verbose)
        end_time = time.time()
        print("Fitting is complete")
        
        # Set attributes based on Frank-Wolfe results
        self.subset_list = result.models
        self.niter_list = result.niters
        self.run_time = end_time - start_time

        return





