# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
from sklearn import metrics
from bisect import bisect_left
import json

with open("DBTSampleSet", "r") as fp:
    DBT_samples = json.load(fp)
    
    
class ScoringM:
    def __init__(self, pi, DBTSampleSet=DBT_samples, curve='prc'):
        self.pi = pi
        self.DBTSampleSet = DBTSampleSet
        self.curve = curve
        self.CurveSet = self.SimulateCurves()
        self.AUCSet = self.SimulateAUC()
        self.Y_set = None
        
    def phi_x(self, alpha, beta):
        if self.curve == 'prc':
            return 1-beta
        if self.curve == 'lift':
            return self.pi*(1-beta) + (1-self.pi)*alpha
        
    def phi_y(self, alpha, beta):
        percentage = self.pi*(1-beta) + (1-self.pi)*alpha
        if self.curve == 'prc':
            return self.pi*(1-beta) / percentage
        if self.curve == 'lift':
            return (1-beta) / percentage
        
    def SimulateCurves(self):
        if self.curve == 'prc':
            y0 = 1
        if self.curve == 'lift':
            y0 = 1/self.pi
            
        samples = []
        for tree in self.DBTSampleSet:
            simulated_curve = {}
            alpha = np.array(tree['alpha'])
            beta = np.array(tree['beta'])            
            simulated_curve['lx'] = self.phi_x(alpha, beta)
            simulated_curve['ly'] = np.insert(self.phi_y(alpha[1:], beta[1:]), 0, y0)
            samples.append(simulated_curve)
            
        return samples
    
    def SimulateAUC(self):
        samples = []
        for l in self.CurveSet:
            x = l['lx']
            y = l['ly']
            auc = np.sum(y[1:] * (x[1:] - x[:-1]))
            samples.append(auc)
            
        return np.array(samples)
    
    def O_AUC(self, x):
        return np.sum(self.AUCSet < x) / self.AUCSet.size
    
    def SimulateY(self, X):
        samples = []; i = 0
        for l in self.CurveSet:
            tree = self.DBTSampleSet[i]
            j = bisect_left(l['lx'], X)
            x1 = l['lx'][j-1]
            x2 = l['lx'][j]
            
            if self.curve == 'prc':
                alpha1 = tree['alpha'][j-1]
                alpha2 = tree['alpha'][j]
                alpha = alpha1 + (alpha2-alpha1) * (X-x1) / (x2-x1)
                Y = self.pi*X / (self.pi*X + (1-self.pi)*alpha)
            
            if self.curve == 'lift':
                beta1 = tree['beta'][j-1]
                beta2 = tree['beta'][j]
                beta = beta1 + (beta2-beta1) * (X-x1) / (x2-x1)
                Y = (1-beta) / X
            
            samples.append(Y)
            i += 1
            
        self.Y_set = np.array(samples)
        
    def O_Y(self, y):
        return np.sum(self.Y_set < y) / self.Y_set.size
    
    
# in terms of e1, e2
def f1_line(f1, e1_list, pi=0.5):
    delta = 2*pi-1
    e2_list=[]
    for e1 in e1_list:
        lamda = 1 - e1
        lamda_pp = (2 * f1 - (1-delta) * f1 * lamda) /((1+delta)*(2-f1))
        e2_list.append(1-lamda_pp)
    
    return e2_list

def O_f1(f1, pi=0.5):
    delta = 2*pi-1
    auc = (f1*(3+delta)) / (2*(1+delta)*(2-f1))
    if f1*(3+delta) <= 2*(1+delta):
        return auc
    else:
        return auc - (f1*(3+delta)-2*(1+delta))**2 / (2*f1*(2-f1)*(1-delta**2))
    
# in terms of errors
def MCC_line(mcc, e1_list, pi=0.5):
    delta = 2*pi-1
    e2_list=[]
    a_p=1 + mcc**2 * (1+delta) / (1-delta)
    a_n=1 + mcc**2 * (1-delta) / (1+delta)
    b_p=1 + mcc**2 * delta / (1-delta)
    
    b_n=1 - mcc**2 * delta / (1+delta)
    c = 1 - mcc**2
    for e1 in e1_list:
        ln = 1 - e1
        lp = (b_p-c*ln)/a_p + np.sqrt(((b_p-c*ln)/a_p)**2 - (a_n*ln**2 - 2*b_n*ln + c)/a_p)
        e2_list.append(1-lp)
    return e2_list

def CR_MCC(mcc, pi=0.5):
    delta = 2*pi-1
    n_dec = 200
    a_n = 1 + mcc**2 * (1-delta) / (1+delta)
    a_p = 1 + mcc**2 * (1+delta) / (1-delta)
    ln_0 = 2*mcc**2 / (a_n*(1+delta))
    lp_0 = 2*mcc**2 / (a_p*(1-delta))
    lam_nn = (1-ln_0)*(np.arange(n_dec)+1)/n_dec + ln_0
    e1_list = 1 - lam_nn
    height = ((1-lp_0)/2 - sum(MCC_line(mcc=mcc, e1_list=e1_list.tolist(), pi=pi)) + n_dec) / n_dec
    return (1-ln_0)*height + ln_0
    
def O_MCC(mcc, pi=0.5):
    if mcc < 0:
        return 1-CR_MCC(-mcc, pi)
    elif mcc == 0:
        return np.float64(0.5)
    else:
        return CR_MCC(mcc, pi)
    