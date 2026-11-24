import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from objects import *
import const

class Gauss_Legendre:
    '''
        This class hlds the implementation of Gauss Legendre quadrature.
        Supports quadratures with 2 or 3 points and functions ranging from
        [-1,1]
    '''
    points_2 = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
    Ak_2 = np.array([1.0, 1.0])

    points_3 = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    Ak_3 = np.array([5/9, 8/9, 5/9])

    @classmethod
    def _rule(cls, n_points_quad: int):
        '''
            Returns (points, weights) of the rule with n_points_quad points.
        '''
        if n_points_quad == 2:
            return cls.points_2, cls.Ak_2
        if n_points_quad == 3:
            return cls.points_3, cls.Ak_3
        raise ValueError(f"Gauss_Legendre supports 2 or 3 points, got {n_points_quad}.")

    @classmethod
    def integrate(cls, f, n_points_quad: int = 3) -> float:
        '''
            Integrate f using n_points_quad
        '''
        points, weights = cls._rule(n_points_quad)
        return np.dot(weights, np.array([f(p) for p in points]))

    @classmethod
    def integrate_vector(cls, f, n_points_quad: int = 3) -> float:
        '''
            Integrate f using n_points_quad. f must return a 1 x num_points_used array where each collumn
            is the function in a point.
        '''
        points, weights = cls._rule(n_points_quad)
        return np.sum(np.dot(weights, np.vstack([f(p) for p in points])))
