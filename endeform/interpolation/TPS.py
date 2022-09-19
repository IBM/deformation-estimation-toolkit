import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

# from scipy.linalg import solve as linsolve
from .I_BaseClasses import BaseInterpolator
from ..helpers.utils import extract_submatrix


# TODO: implement copy() functionality
class TPS(BaseInterpolator):
    """TPS interpolation
    TODO: Documentation
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.coef_ = None
        self.knots_ = None
        self.n_ = None
        self.nchannels_ = None

    def fit(self, X, y, alpha=None, check_finite=True):
        """TODO: Docstring

        Parameters
        ----------
        X : np.array
            Shape is (num_knots, self.n_)
        y : np.array
            Shape is (num_knots, self.nchannels_). If y is 1-D, a singleton 2nd dimension is added.
        alpha : float, optional
            The regulatization strength, by default `self.alpha`. `self.alpha` will _not_ be overwritten.
        indices : List of int or slice or any other valid numpy indexing expression, optional
            If precomputed RBFs are used, then these are the indices of the subset of precomputed knots to
            be used in this call to `fit()` (and the subsequent `eval()`)
        """
        if alpha is None:
            alpha = self.alpha

        if y.ndim == 1:
            y = y[:, np.newaxis]
        self.nchannels_ = y.shape[1]

        self.nknots_, self.n_ = X.shape  #  TPS v0 doesn't have nknots_
        self.knots_ = np.hstack((X, y))
        # This solves [ RBF+alpha*I, P; P^T, 0 ]*coef_ = [y; 0]:
        self.coef_ = np.linalg.solve(
            np.concatenate(
                (
                    np.concatenate(
                        (
                            self._generate_rbf_matrix()
                            + alpha * X.shape[0] * np.eye(X.shape[0]),
                            np.hstack((np.ones((X.shape[0], 1)), X)),
                        ),
                        axis=1,
                    ),
                    np.concatenate(
                        (
                            np.vstack((np.ones((1, X.shape[0])), X.T)),
                            np.zeros((self.n_ + 1, self.n_ + 1)),
                        ),
                        axis=1,
                    ),
                ),
                axis=0,
            ),
            np.concatenate((y, np.zeros((self.n_ + 1, self.nchannels_))), axis=0),
        )
        #    check_finite=check_finite, # True is the default, but False may enhance performance
        #    assume_a='sym'  #  according to [1], page 16, A is symmetric but not definite
        #   np.linalg.solve doesn't allow (as of version X) to add any assumptions on the lhs matrix
        #   Besides, tests reveal that assuming 'sym' makes the scipy function _slower_, see also issue #5
        # The long concatenate call is equivalent to
        # np.block( [ [U+alpha*n*I, [1,X] ], [ [1;X.T], 0] ])
        # but roughly 1.5x as fast -- with X.shape[0]=100, it saves about 12us

        return self

    def _generate_rbf_matrix(self, Q=None, pre_knots=None):
        """Computes the radial basis function values
            `R[i,j] = RBF(Q[i,:], knots[j,:])`
            there are 4 different scenarios:
            * `Q = pre_knots = None` : The pair-wise distance between `self.knots_` (as
                 supplied to `fit()`) are computed
            * `Q not None`, `pre_knots = None` : Distances between `Q` and `self.knots_`
            * `Q` and `pre_knots` not `None`: `pre_knots` is used in place of `self.knots_`.
                This would be used to precompute query RBF values and later index instead of recompute.
            * `Q == None`, `pre_knots` not `None`: pairwise distance between pre_knots, which is
                used to precompute the RBF among the knot points.
        NOTE: TPS v0 has no pre_knots argument because it is not needed, but this slightly
        more general implementation (one extra if statement) allows inheritance

        Parameters
        ----------
        Q : np.array or None
            The points which make up the first argument of the RBFs
        pre_knots : np.array or None
            See docstring above

        If no query points are given, then on the pairwise distance among the knots.
        Else on the pairwise difference between query points in Q and the knots
        Returns:
        --------
        np.array :
            Shape (Q.shape[0], self.nknots_) or (self.nknots_, self.nknots_) or
            (Q.shape[0], pre_knots.shape[0]) or (pre_knots.shape[0], pre_knots.shape[0]).

        NOTE: Chance for performance improvement: squareform(pdist(X)) is only faster than cdist(X,X)
        for X.shape[0]>100 (roughly). pdist(X) alone is always the fastest, but yields only a vector
        corresponding to the upper triang entries of a symmetric matrix.
        """
        # scaling, to avoid overflows in r**r
        # uses r^2 log(r) = r log(r^r) = r0 r log(r^(r/r0))
        scale1 = 2  #  worst-case power is now max(r)**scale1
        if Q is None:
            if pre_knots is None:
                r = pdist(self.knots_[:, : self.n_])
                r0 = np.max(r) / scale1
                res = squareform(r0 * r * np.log(r ** (r / r0)))
            else:
                r = pdist(pre_knots)
                r0 = np.max(r) / scale1
                res = squareform(r0 * r * np.log(r ** (r / r0)))
        else:
            if pre_knots is None:
                r = cdist(Q, self.knots_[:, : self.n_])
                r0 = np.max(r) / scale1
                res = r0 * r * np.log(r ** (r / r0))
            else:
                r = cdist(Q, pre_knots)
                r0 = np.max(r) / scale1
                res = r0 * r * np.log(r ** (r / r0))
        return res

    def eval(self, query_points):
        """Evaluate the TPS on a set of query points

        Parameters
        ----------
        query_points : np.array of size N x self.n_
            Each row corresponds to one point at which the TPS is to be evaluated
            If a precomputed RBF is used, `query_points` is optional (and will be ignored if
            a precomputed RBF is already available).
        Returns
        -------
        np.array
            Each row i corresponds to the TPS evaluated at `query_points[i,:]`
        """
        if self.coef_ is None:
            raise RuntimeError(
                "You need to call fit() before trying to evaluate the TPS"
            )

        query_points = np.atleast_2d(query_points)
        return (
            self.coef_[-3, ...]
            + query_points @ self.coef_[-2:, ...]
            + self._generate_rbf_matrix(Q=query_points) @ self.coef_[:-3, ...]
        )

    #  def fit_and_eval(self, coords, values, query_points, alpha=None):
    #  """TODO: Is there any time to be saved by having a single call to _generate_rbf_matrix
    #     instead of one in the fit() method (with the nodes,to compute the lhs matrix) and then
    #     again in the eval method (with the query points)?
    #  """
    #    if alpha is None: alpha = self.alpha


class TPSprecompute(TPS):
    def __init__(self, alpha):
        super().__init__(alpha=alpha)
        self._pre_query_RBF = None
        self._pre_query_points = None
        self._pre_knot_RBF = None
        self._pre_knots = None

    def fit(self, knot_indices, y, alpha=None, check_finite=True):
        """TODO: Docstring

        Parameters
        ----------
        knot_indices : List of int or slice or any other valid numpy indexing expression
            Typically, only a subset of the knots for which the RBF has been precomputed will have to be used
            in each iteration. Specify the indices of this subset here. Use `Ellipsis` to use all the knots.
            The indexing should result in `num_knots` elements.
        y : np.array
            Shape is (num_knots, self.nchannels_). If y is 1-D, a singleton 2nd dimension is added.
        alpha : float, optional
            The regulatization strength, by default `self.alpha`. `self.alpha` will _not_ be overwritten.
        """
        # did we precompute before?
        if self._pre_knot_RBF is None:
            # no, so error
            raise RuntimeError(
                "You need to call precompute_RBF() with some pre_knots before calling fit()"
            )

        if alpha is None:
            alpha = self.alpha

        if y.ndim == 1:
            y = y[:, np.newaxis]
        self.nchannels_ = y.shape[1]
        self.n_ = self._pre_knots.shape[1]

        if knot_indices is Ellipsis:
            self.nknots_ = self._pre_knots.shape[0]
            self._knot_RBF = self._pre_knot_RBF
        else:
            # TODO: works only when idexing with list, see my stackoverflow question for more
            #       general solutions https://stackoverflow.com/q/68180645/14477568
            self.nknots_ = len(knot_indices)
            self._knot_RBF = extract_submatrix(
                self._pre_knot_RBF,
                knot_indices,
                knot_indices,
                self._pre_knots.shape[0],
                self.nknots_,
                self.nknots_,
            )
        self.knot_indices = knot_indices
        self.knots_ = self._pre_knots[knot_indices, :]

        self.coef_ = np.linalg.solve(
            np.concatenate(
                (
                    np.concatenate(
                        (
                            self._knot_RBF
                            + alpha * self.nknots_ * np.eye(self.nknots_),
                            np.hstack(
                                (
                                    np.ones((self.nknots_, 1)),
                                    self._pre_knots[knot_indices, :],
                                )
                            ),
                        ),
                        axis=1,
                    ),
                    np.concatenate(
                        (
                            np.vstack(
                                (
                                    np.ones((1, self.nknots_)),
                                    self._pre_knots[knot_indices, :].T,
                                )
                            ),
                            np.zeros((self.n_ + 1, self.n_ + 1)),
                        ),
                        axis=1,
                    ),
                ),
                axis=0,
            ),
            np.concatenate((y, np.zeros((self.n_ + 1, self.nchannels_))), axis=0),
        )
        return self

    def eval_pre(self, query_indices=None):
        """Evaluate the TPS on a set of query points

        Parameters
        ----------
        query_indices :
            if not all query points for which the RBF was precomputed are to be used then supply
            the indices of the ones you want to use here
        Returns
        -------
        np.array
            Each row i corresponds to the TPS evaluated at `query_points[i,:]`
        """
        if self.coef_ is None:
            raise RuntimeError(
                "You need to call fit() before trying to evaluate the TPS"
            )

        # did we precompute before?
        if self._pre_query_RBF is None:
            raise RuntimeError(
                "You need to call precompute_RBF with query_points before calling eval()"
            )

        if (query_indices is None) or (query_indices is Ellipsis):
            # in this case we can use basic indexing
            return (
                self.coef_[-3, ...]
                + self._pre_query_points @ self.coef_[-2:, ...]
                + self._pre_query_RBF[:, self.knot_indices] @ self.coef_[:-3, ...]
            )
        elif self.knot_indices is Ellipsis:
            # still can use basic indexing
            return (
                self.coef_[-3, ...]
                + self._pre_query_points[query_indices, :] @ self.coef_[-2:, ...]
                + self._pre_query_RBF[query_indices, :] @ self.coef_[:-3, ...]
            )
        else:
            # have to extract a submatrix
            return (
                self.coef_[-3, ...]
                + self._pre_query_points[query_indices, :] @ self.coef_[-2:, ...]
                + extract_submatrix(
                    self._pre_query_RBF,
                    query_indices,
                    self.knot_indices,
                    self._pre_knots.shape[0],
                    len(query_indices),
                    self.nknots_,
                )
                @ self.coef_[:-3, ...]
            )

    def precompute_RBF(self, *, pre_query_points=None, pre_knots=None):
        """Precomputes the RBF matrix for the knots and the query points given in `X`."""
        if (pre_knots is None) and (pre_query_points is None):
            raise ValueError(
                "You need to specify at least one of `query_points` and `pre_knots`."
            )
        if pre_knots is not None:
            self._pre_knot_RBF = self._generate_rbf_matrix(pre_knots=pre_knots)
            self._pre_knots = pre_knots
        if pre_query_points is not None:
            if pre_knots is None:
                if self._pre_knots is not None:
                    pre_knots = self._pre_knots
                else:
                    raise RuntimeError(
                        "If calling with only `pre_query_points`, `_pre_knots` must have been set already\
                        e.g. by previous call to `fit()`."
                    )
            pre_query_points = np.atleast_2d(pre_query_points)
            self._pre_query_RBF = self._generate_rbf_matrix(
                Q=pre_query_points, pre_knots=pre_knots
            )
            self._pre_query_points = pre_query_points
        return self._pre_knot_RBF, self._pre_query_RBF

    # vvv Should be able to inhehrit this from TPS!
    # def _generate_rbf_matrix(self, Q=None, pre_knots=None):
    #     """Computes the radial basis function values
    #         `R[i,j] = RBF(Q[i,:], knots[j,:])`
    #         there are 4 different scenarios:
    #         * `Q = pre_knots = None` : The pair-wise distance between `self.knots_` (as
    #              supplied to `fit()`) are computed
    #         * `Q not None`, `pre_knots = None` : Distances between `Q` and `self.knots_`
    #         * `Q` and `pre_knots` not `None`: `pre_knots` is used in place of `self.knots_`.
    #             This would be used to precompute query RBF values and later index instead of recompute.
    #         * `Q == None`, `pre_knots` not `None`: pairwise distance between pre_knots, which is
    #             used to precompute the RBF among the knot points.

    #     Parameters
    #     ----------
    #     Q : np.array or None
    #         The points which make up the first argument of the RBFs
    #     pre_knots : np.array or None
    #         See docstring above

    #     If no query points are given, then on the pairwise distance among the knots.
    #     Else on the pairwise difference between query points in Q and the knots
    #     Returns array of shape (Q.shape[0], self.knots_shape[0])

    #     NOTE: Chance for performance improvement: squareform(pdist(X)) is only faster than cdist(X,X)
    #     for X.shape[0]>100 (roughly). pdist(X) alone is always the fastest, but yields only a vector
    #     corresponding to the upper triang entries of a symmetric matrix.
    #     """
    #     # scaling, to avoid overflows in r**r
    #     # uses r^2 log(r) = r log(r^r) = r0 r log(r^(r/r0))
    #     scale1 = 2  #  worst-case power is now max(r)**scale1
    #     if Q is None:
    #         if pre_knots is None:
    #             r =  pdist(self.knots_[:, :self.n_])
    #             r0 = np.max(r)/scale1
    #             res = squareform(r0 * r * np.log( r**(r/r0) ))
    #         else:
    #             r =  pdist(pre_knots)
    #             r0 = np.max(r)/scale1
    #             res = squareform(r0 * r * np.log( r**(r/r0) ))
    #     else:
    #         if pre_knots is None:
    #             r = cdist(Q, self.knots_[:, :self.n_])
    #             r0 = np.max(r)/scale1
    #             res = r0 * r * np.log(r**(r/r0) )
    #         else:
    #             r = cdist(Q, pre_knots)
    #             r0 = np.max(r)/scale1
    #             res = r0 * r * np.log(r**(r/r0) )
    #     return res
