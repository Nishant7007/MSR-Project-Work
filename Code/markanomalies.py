#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:53:41 2020

@author: ictd
"""

from loadSeries import *
from datetime import timedelta

def find_date(df,start,end):
    print(df[start:end].idxmax())
    print(df[start:end].min())
    print('start:',df[start:end].idxmin()-timedelta(21))
    print('end:',df[start:end].idxmin()+timedelta(21))
    print('--------------------')


lasalgaon_mandi.index=pd.date_range('2006-01-01',periods=len(lasalgaon_mandi))
bangalore_mandi.index=pd.date_range('2006-01-01',periods=len(bangalore_mandi))
azadpur_mandi.index=pd.date_range('2006-01-01',periods=len(azadpur_mandi))



end='2019-01-12'
start=pd.to_datetime(end)-timedelta(14)


find_date(lasalgaon_mandi,start,end)
find_date(bangalore_mandi,start,end)
find_date(azadpur_mandi,start,end)