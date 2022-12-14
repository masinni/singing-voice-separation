'''
Archetypal analysis with sparseness constraints decomposes a matrix X
that is the magnitude spectrogram of a mixed signal,
into a low rank matrix X*C*S that represents the accompaniment music part
and a sparse matrix E that represents the singing voice part.

Matrices C, S have dimentions: C = m*k,  S =k*m,
where k = number of archetypes and m = number of columns of matrix X.
'''

import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize


class Archetypal_analysis_sparseness():
    def __init__(self, X, k, lmbda, max_iter, wav_name):
        """
        Class initialization
        -------------
        param X : Data matrix
        param k : Number of archetypes
        param lmbda : trade-off parameter
        param tol : stopping criterion

        Attributes
        -------------
        C : matrix of basis vectors
        S : matrix of coefficients
        X*C*S : Low-rank matrix
        E : Sparse matrix
        frob_error : Error of reconstruction
        c_err : Error of reconstruction of matrix C
        s_err : Error of reconstruction of matrix S
        e_err : Error of reconstruction of matrix E
        """
        self.X = X
        self.n, self.m = self.X.shape
        self.k = k
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.wav_name = wav_name
        self.iter = 1
        self.tol = 1e-2

    def convergence_criterion(self):
        """
        Reconstruction error between X and (X*C*S+E)
        """
        if hasattr(self, 'C') and hasattr(self, 'S') and hasattr(self, 'E'):
            XCS = (self.X).dot(self.C).dot(self.S)
            error = norm(self.X - XCS - self.E, 'fro') / norm(self.X, 'fro')
        else:
            error = None
        return error

    def conv_crit_individual(self, prevC, prevS, prevE):
        """
        Individual error of every matrix for convergence guarantee
        """
        c_err = norm(self.C-prevC, 'fro') / norm(self.X, 'fro')
        s_err = norm(self.S-prevS, 'fro') / norm(self.X, 'fro')
        e_err = norm(self.E-prevE, 'fro') / norm(self.X, 'fro')
        return c_err, s_err, e_err

    def initialize_c(self):
        """
        Initalize C to random, non-negative values which sum to one.
        """
        avg = np.sqrt(self.X.mean() / self.k)
        rng = np.random.RandomState(13)
        self.C = avg * rng.randn(self.m, self.k)
        np.abs(self.C, out=self.C)
        self.C = self.C/self.C.sum(axis=0, keepdims=1)

    def initialize_s(self):
        """
        Initalize S to random, non-negative values which sum to one
        """
        avg = np.sqrt(self.X.mean() / self.k)
        rng = np.random.RandomState(12)
        self.S = avg * rng.randn(self.k, self.m)
        np.abs(self.S, out=self.S)
        self.S = self.S/self.S.sum(axis=0, keepdims=1)

    def initialize_e(self):
        """
        Initalize sparse matrix E with zero values
        """
        self.E = np.zeros((self.n, self.m))

    def update_C(self):
        """
        C update rule, considering all columns sum to 1
        """
        prevC = self.C.copy()
        c_num = (self.X.T).dot(self.X).dot(self.S.T)
        c_den = (self.X.T).dot(self.X).dot(self.C).dot(self.S).dot(self.S.T) + \
                (self.X.T).dot(self.E).dot(self.S.T)
        C = (prevC * c_num) / c_den
        C = normalize(C, norm='l1', axis=0)
        self.C = C

        return self.C, prevC

    def update_S(self):
        """
        S update rule, considering all columns sum to 1
        """
        prevS = self.S.copy()
        s_num = (self.C.T).dot(self.X.T).dot(self.X)
        s_den = s_num.dot(self.C).dot(self.S) + \
            (self.C.T).dot(self.X.T).dot(self.E)
        S = (prevS*s_num) / s_den
        S = normalize(S, norm='l1', axis=0)
        self.S = S

        return self.S, prevS

    def update_E(self):
        """
        Update E using shrinkage operator
        """
        prevE = self.E.copy()

        Eraw = self.X - (self.X).dot(self.C).dot(self.S)
        E_upd = np.maximum(Eraw - self.lmbda, 0) + np.minimum(Eraw + self.lmbda, 0)
        self.E = np.maximum(E_upd, 0.0)

        return self.E, prevE

    def compute_factors(self):

        self.initialize_c()
        self.initialize_s()
        self.initialize_e()

        self.frob_error = np.zeros(self.max_iter)

        c_err = np.zeros(self.max_iter)
        s_err = np.zeros(self.max_iter)
        e_err = np.zeros(self.max_iter)

        for i in range(self.max_iter):

            # Update matrices C, S, E
            self.C, prevC = self.update_C()
            self.S, prevS = self.update_S()
            self.E, prevE = self.update_E()

            # Compute reconstruction error
            self.frob_error[i] = self.convergence_criterion()

            # Compute matrices' convergence error
            c_err[i], s_err[i], e_err[i] = self.conv_crit_individual(
                                           prevC,
                                           prevS,
                                           prevE)
            max_error_list = [c_err[i], s_err[i], e_err[i]]

            """
            diff parameter shows how the error of reconstruction changes as
            the repetitions increase
            max_err parameter shows the maximum error out of the three matrices
            reconstruction error
            Stopping criteria:
            """
            diff = abs(self.frob_error[i-1] - self.frob_error[i])
            max_err = max(max_error_list)
            e = 1e-3
            if diff <= self.tol and max_err < e:
                print(self.wav_name,
                      '-- iter: ', self.iter,
                      ', max error: ', max_err)
                break

            self.iter += 1

        # Low rank matrix XCS:
        XCS = (self.X).dot(self.C).dot(self.S)

        return XCS, self.E

    def matrix_formulation(self):

        XCS, E = self.compute_factors()
        print('------------------------------------------------')

        return XCS, E
