from objects import *
from quadrature import *
from liquid import *
from solve import *
import const

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import floor
 
class Result:
    '''
        Class that holds the information of a PVI solution
    '''
    def __init__(self, qnt_passos, h, u, t):
        self.qnt_passos = qnt_passos #Amount of steps executed
        self.h = h #Step of t
        self.u = u #Vector with the solutiom
        self.t = t #Vector with the values of time used. Note that  t_final <= t[-1] < t_final + h

def calc_amount_steps(t0, t_final, h):
    tam_intervalo = t_final - t0

    qnt_passos = tam_intervalo/h
    if qnt_passos != floor(qnt_passos): qnt_passos = floor(qnt_passos) + 1
    else: qnt_passos = int(qnt_passos)
    qnt_passos += 1 #Add u0 to the solution vector
    return qnt_passos

def initialize_method(qnt_passos, t0, u0):
    u = np.zeros((len(u0), qnt_passos)) #Creates a matrix in which the collumns are the solution in a determined time 
    #and the lines are the solution of a variable across time
    t = np.zeros(qnt_passos)

    u[:, 0] = u0
    t[0] = t0
    return u, t

def euler_explicito(t0, t_final, h, u0: np.array, f):
    qnt_passos = calc_amount_steps(t0, t_final, h)
    u, t = initialize_method(qnt_passos, t0, u0)

    for i in range(1,qnt_passos):
        t[i] = t[i-1] + h
        u[:, i] = u[:, i-1] + h*f(u[:, i-1], t[i-1])

    return Result(qnt_passos, h, u, t)
     

def RK4(t0, t_final, h, u0: np.array, f):
    qnt_passos = calc_amount_steps(t0, t_final, h)
    u, t = initialize_method(qnt_passos, t0, u0)

    for i in range(1,qnt_passos):
        t[i] = t[i-1] + h
        K1 = f(u[:, i-1], t[i-1])
        K2 = f(u[:, i-1] + (h/2)*K1, t[i-1] + h/2)
        K3 = f(u[:, i-1] + (h/2)*K2, t[i-1] + h/2)
        K4 = f(u[:, i-1] + h*K3, t[i-1] + h)
        u[:,i] = u[:,i-1] + (h/6)*(K1 + 2*K2 + 2*K3 + K4)

    return Result(qnt_passos, h, u, t)
    