#
# Copyright 2020- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
#
from abc import ABC, abstractmethod

from numpy import meshgrid, column_stack

from endeform.helpers.plot_helpers import rect_grid


class BaseInterpolator(ABC):
    """The base class for interpolators"""

    @abstractmethod
    def fit(self, coords, values, **kwargs):
        """Computes an interpolation function F so that
        F(`coords[i,:]`) is close to `values[i,:]`.

        Returns
        -------
        self
        """
        return NotImplemented

    @abstractmethod
    def eval(self, query_points, **kwargs):
        """Evaluates the interpolator (after it's been fitted) at `query_points`, so it should
        return an array V such that V[i, :]=F(`query_points[i,:]`)
        """
        return NotImplemented

    def fit_and_eval(self, coords, values, query_points, **kwargs):
        """fit to `coord` and `values, then evaluate at `query_points` and return result.
        Typically and if possible, you should overload this with something that is more
        efficient than fitting followed by evaluating."""
        self.fit(coords, values, **kwargs)
        return self.eval(query_points, **kwargs)

    def draw_warped_grid(
        self,
        wh,
        skip=(20, 20),
        ax=None,
        draw_frame=True,
        frame_linestyle="k+--",
        **kwargs
    ):
        """Warp a regular rectangular grid and display the warped grid.
        The lines at 0 and at w, resp. h, are always drawn.
        NOTE: This works only for TPS with 2 channels, i.e. with nchannels_==2
        NOTE: if the grid is very fine, this might take a long time.

        Parameters
        ----------
        wh : tuple of 2 int
            (width, height) of the grid)
        skip : tuple of 2 int, optional
            skip `skip[0]` steps in x-direction before drawing another grid
             line, analogous in y-direction, by default (20,20)
        ax : plt.Axis, optional
            Axis to draw into; if None, the current axis is used, and if there isn't one,
             then a new one is created, by default None
        draw_frame : bool, optional
            Whether to draw the recangle [0,w] x [0,h], by default True
        frame_linestyle : str, optional
            Linestyle of the frame, if it is drawn, by default 'k+--'
        remaining `**kwargs` are passed through to endeform.helpers.plot_helpers.rect_grid
        Returns
        -------
        plt.Axis
            The axis that was drawn into
        """
        w, h = wh
        # generate the ranges and add the last element, if necessary
        yrange = list(range(0, h, skip[1]))
        if not yrange[-1] == h:
            yrange += [h]
        xrange = list(range(0, w, skip[0]))
        if not xrange[-1] == w:
            xrange += [w]
        Y, X = meshgrid(yrange, xrange)
        Z = self.eval(column_stack((X.flat, Y.flat))).reshape(
            (X.shape[0], X.shape[1], 2)
        )
        ax = rect_grid(Z[:, :, 1].T, Z[:, :, 0].T, ax=ax, **kwargs)
        if draw_frame:
            ax.plot([0, 0, w, w, 0], [0, h, h, 0, 0], frame_linestyle, linewidth=1)
        return ax
