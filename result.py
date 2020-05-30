"""
SGD
"""
import numpy as np
import matplotlib.pyplot as plt
import csv


df7 = csv.reader(open('0526closex-data.csv', 'r'))
data7 = [ v for v in df7]
mat7 = np.array(data7)
mat8 = mat7[1:]
"""下でX_dataひっくり返した                   # 見出し行を外す
だかlogとって100倍した20204/23と22の差が一番上のデータ
"""
x_data_pre =np.flipud(mat8[:, 1:].astype(np.float))
print('x_data_pre.shape=', x_data_pre.shape)



"""
ここから具体的な実験用データつくる　列でひとつ隣の列に一日前のdataを書いていくそんで、最後の4行は平均線
"""
data_predic = x_data_pre.T
print('data_predic.shape=', data_predic.shape)
#何日前までのデータが必要かを合わせる
day_ago=6811
data_pre=data_predic[:,:day_ago-1]
print('data_pre.shape=', data_pre.shape)

data_preave = np.array(np.append(data_pre,[0.77425865,0.331625257,-0.11798576,-0.012014351])).reshape(1,day_ago+3)
#頭に後
data_preav2=np.append(2.374244032,data_preave).reshape(1,day_ago+4)
print('data_preav2.shape=', data_preav2.shape)



"""これをコンソールで実行
network.predict(data_preav2) 
""" 