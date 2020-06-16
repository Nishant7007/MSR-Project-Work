#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:35:24 2020

@author: ictd
"""


import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

from loadSeries import *

dummy=azadpur_mandi['price']/azadpur_mandi_lstm['price']

lstm=bangalore_mandi_arima['price']*.3+bangalore_mandi_sarima['price']*.35+azadpur_mandi_arimax2['price']*.25+azadpur_mandi_sarimax['price']*.1
#lstm=lstm*dummy
print(lstm)

lstm.to_csv('bangalore_mandi_lstm.csv')