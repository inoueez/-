#Name （neural_SGD.py）
SGD、確率的勾配降下法とよばれる機械学習で使われる最も一般的な学習率で(pythonで始める機械学習１利用）日経平均が次の日に1%以下で上がるか、1%以下で下がるか、それともそれ以上、もしくはそれ以下で動くかを4値分析を用いて結果を出す。

# 特徴

ネットではkaras、tensorflowを用いたものなどは存在しているが、そのため開発環境に依存している。一方こちらはkarasなど深層学習の開発pakageをほぼ用いないためエラーの原因がバージョンなどということは少ないと考えられる。また、テクニカル指標を組み込んでみようと、平均線などを導入した。

# 開発環境
とりあえず、自分の開発環境

* python 3.7
* anaconnnda 3
*spyder
# インストール

anaconda3、pythonのインストールを行う

必要なものはnumpy、 matplotlib、csv、timeなので
anaconda promptを開いて

```bash
conda install numpy
```
"""
numpyのところにmatplotlibなど名前変えていれる



# Name（result.py）
 
上のnural_SGD.pyを実行した後、これを実行することで、実際に次の日の4値の結果が表示される 



# Name（0526closex-data.csv）

読み込みのデータ1 

一列目に日付、二列目に前日との差を対数表時にしたもの。



# Name（0526closet-data.csv）
読み込みデータ２

一列に日付その後は次の日の4値分類した結果


# Name（teqniq-diff.csv)
読み込みデータ3

一列目日付
その後に5日平均線、25日平均線、75日平均線,200日平均線を組み込む




# Author

* 作成者
Inoue akito
* E-mail
hutatsu0725@gmail.com
