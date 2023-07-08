#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 13:57:05 2023

@author: xuxian
"""
import numpy as np

data = np.loadtxt('phase_LL_wfn_allncnv_ex1.dat')

data2 = data[0:-1:2, :]

np.savetxt('phase_LL_wfn_allncnv_ex1_odd.dat',data2)