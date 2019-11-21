#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, RobustScaler


df = pd.read_csv('creditcard.csv')

normAmount = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

df2 = pd.DataFrame(normAmount, columns=['normAmount'])
    
#df.append(df2)
#df = df.drop(['Time','Amount'],axis=1)
df2.to_csv(r'/Users/apple/Documents/SEM2_2019/research/ResearchProject_SEM1_Riris/REPEN/normAmount.csv', index=False)
