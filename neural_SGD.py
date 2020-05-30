"""
SGD  数字が近いときと、0.5以下の時外す
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD
import time
t1 = time.time() 
 


#まずはdata作成から    1995/1/6から2020/4/23の前日との差13604列

df1 = csv.reader(open('0526closex-data.csv', 'r'))
data1 = [ v for v in df1]
mat = np.array(data1)
mat2 = mat[1:]    
"""下でX_dataひっくり返した                   # 見出し行を外す
だかlogとって100倍した20204/23と22の差が一番上のデータ
"""
x_data =np.flipud(mat2[:, 1:].astype(np.float))
print('x_data.shape=', x_data.shape)
# ラベルデータ
# 1％以上／0％以上／-1％以上／-1％未満
df2 = csv.reader(open('0526closet-data.csv', 'r'))
data2 = [ v for v in df2]
mat3 = np.array(data2)
mat4 = mat3[1:]                       # 見出し行を外す
t_data =np.flipud(mat4[:, 1:].astype(np.float)) # 2列目以降を抜き出してfloat変換
print('t_data.shape=', t_data.shape)


"""
diff 平均線の組み込み   5,25,75,200diffの平均線で4列　
200日前までのデータがない過去を取り出したので13405行  過去削ったから13604問題なし
"""
df3=csv.reader(open('teqni_diff0526.csv', 'r'))
data3 = [ v for v in df3]
mat4 = np.array(data3)
mat5 = np.flipud(mat4[1:,2:])   
xave_data= mat5[:len(mat5)-200].astype(np.float)
print('xave_data.shape=', xave_data.shape)

"""
ここから具体的な入力と教師用データつくる　列でひとつ隣の列に一日前のdataを書いていく  上が新しい日付　dataもtargetも
"""
data = []
target = []
#0日前からスタートで6802-1日前,つまり6801日前でうえから当日
day_ago=len(x_data)//2
print('day_ago', day_ago)
for i in range(0, day_ago):
  data.append(x_data[i:i+day_ago, :])
  target.append(t_data[i,:])  #これにより一番上がday_ago前


#50日平均線などを付け足す
xave_data2=xave_data[:day_ago]


x0 = np.array(data).reshape(len(data), day_ago)
#平均線の追加
x=np.append(x0,xave_data2,axis=1)
t = np.array(target).reshape(len(data), 4)
print('x.shape=', x.shape)
print('t.shape=', t.shape)

x_data_test=x[:40,:]
x_data_train=x[40:,:] 
t_data_test=t[:40,:]
t_data_train=t[40:,:] 

print("x_data_train.shape=",x_data_train.shape,
"x_data_test.shape=",x_data_test.shape, 
"t_data_train.shape=",t_data_train.shape,
"t_data_test.shape=",t_data_test.shape)


"""
ニューラルネットワークの環境構築
"""

weight_decay_lambda = 0.1
network = MultiLayerNet(input_size=day_ago+4, hidden_size_list=[100], output_size=len(t[0]),
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01)





#パラメータ
iters_num= 4000 #勾配法の更新の回数　多いほどいい？
train_size=len(t_data_train)  #入力データの行数　サンプル数 testデータと分けたこと忘れずに
batch_size =300      #これがバッチの数#分からんから
learning_rate=0.01

train_loss_list=[]
train_acc_list=[]
test_acc_list=[]
#1エポック当たりの繰り返し数
iter_per_epoch =max(train_size/batch_size,1)


for i in range(iters_num):
    batch_mask = np.random.choice(train_size , batch_size)
    x_batch=x_data_train[batch_mask]
    t_batch=t_data_train[batch_mask]
    #grad=network.numerical_gradient(x_batch,t_batch)
    grad=network.gradient(x_batch,t_batch)
    for key in("W1","b1","W2","b2"): #Wは大文字
        network.params[key] -=learning_rate*grad[key]
        loss=network.loss(x_batch,t_batch)
        train_loss_list.append(loss)
        """
        train_acc = network.accuracy(x_data_train, t_data_train)
        test_acc = network.accuracy(x_data_test, t_data_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        """
print("ここまで１")


"""
network.predict(引数)で　予測が可能　　この引数は入力データの列数と同じであれば行数にかかわらず出てくる

今回は6802やったかなあれを入れたら、onehotの結果が出る


"""

# 正答率、準正答率（騰落）集計
preds =  network.predict(x_data_test)
correct = 0
semi_correct = 0
miss_big = 0
for i in range(len(preds)):
  pred = np.argmax(preds[i,:])
  tar = np.argmax( t_data_test[i,:])
  if pred == tar :
    correct += 1
  else :
    if pred+tar == 1 or pred+tar == 5 :
      semi_correct += 1
    else :
        if abs(pred-tar) ==3:
            miss_big += 1
print("正答率:", 1.0 * correct / len(preds))
print("準正答率（騰落）:", 1.0 * (correct+semi_correct) / len(preds))
print("やっちゃいけないミス:",miss_big/ len(preds))



"""1に近いとどうなるか
"""

cor=0
semi = 0
for i in range(len(preds)):
  pred = np.argmin(np.abs(np.asarray(preds[i,:]) - 1))
  tar  = np.argmin(np.abs(np.asarray(t_data_test[i,:]) - 1)) 
  if pred == tar :
      cor += 1
  else :
    if pred+tar == 1 or pred+tar == 5 :
      semi += 1
print("１に近い正答率:", 1.0 * cor / len(preds))
print("1に近い準正答率（騰落）:", 1.0 * (cor+semi) / len(preds))







#損失関数
plt.xlabel("itenum*4")
plt.ylabel("lost")
plt.plot(train_loss_list)
plt.show


#認証誤差について　訓練データだけに都合のいいこたえじゃないか
plt.figure()
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

"""

４つの平均とって一番遠いやつとｋならどうなるかな
"""











t2 = time.time()
# 経過時間を表示
elapsed_time = t2-t1
print(f"経過時間：{elapsed_time}")

"""答え合わせ分析

for i in range(40):
    print(i)
    print("正解",t_data_test[i])
    print("答え",network.predict(x_data_test)[i])
"""