import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from objects import *
from liquid import *
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
    def integrate(cls, f, n_points_quad: int = 3) -> float:
        '''
            Integrate f using n_points_quad
        '''
        sum = 0.0
        tmp = []
        if n_points_quad == 2:
            for i in range(2): tmp.append(f(cls.points_2[i]))
            tmp = np.array(tmp)
            sum = np.dot(cls.Ak_2, tmp)
        elif n_points_quad == 3:
            for i in range(3): tmp.append(f(cls.points_3[i]))
            tmp = np.array(tmp)
            sum = np.dot(cls.Ak_3, tmp)
        return sum

    @classmethod
    def integrate_vector(cls, f, n_points_quad: int = 3) ->float:
        '''
            Integrate f using n_points_quad. f must return a 1 x num_points_used array where each collumn
            is the function in a point.
        '''
        sum = 0.0
        tmp = []
        if n_points_quad == 2:
            for i in range(2): tmp.append(f(cls.points_2[i]))
            tmp = np.vstack(tmp)
            integral_array = np.dot(cls.Ak_2, tmp)
            sum = np.sum(integral_array)
        elif n_points_quad == 3:
            for i in range(3): tmp.append(f(cls.points_3[i]))
            tmp = np.vstack(tmp)
            integral_array = np.dot(cls.Ak_3, tmp)
            sum = np.sum(integral_array)
        return sum
