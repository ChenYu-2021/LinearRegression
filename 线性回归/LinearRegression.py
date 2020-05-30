import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
# 读取数据,第二列是房价，训练数据要把第二列删除
'''
housing = pd.DataFrame()
with open('kc_train.csv') as f:
    lines = f.readlines()
    for line in lines:
        housing.append(line)
'''

columns_name = ['销售日期', '销售价格', '卧室数', '浴室数', '房屋面积', '停车面积', '楼层数',
                '房屋评分', '建筑面积', '地下室面积', '建筑年份', '修复年份', '纬度', '经度']
test_columns_name = ['销售日期', '卧室数', '浴室数', '房屋面积', '停车面积', '楼层数',
                '房屋评分', '建筑面积', '地下室面积', '建筑年份', '修复年份', '纬度', '经度']
housing = pd.read_csv('kc_train.csv', header=None) # header=None表示不会从文件中指定行作为列名
housing.columns = columns_name

train_data_x = housing.drop(['销售价格'], axis=1)
train_data_y = housing['销售价格']

# 特征缩放（训练数据缩放）
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(train_data_x)
scaler_x = minmax_scaler.transform(train_data_x)
x_scaler = pd.DataFrame(scaler_x, columns=train_data_x.columns)

# 将数据分成训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(x_scaler,train_data_y, test_size=0.05)


# 建立线性回归模型
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()

lr_model.fit(train_x, train_y)

# 测试模型的好坏
from sklearn.metrics import mean_squared_error
predicts = lr_model.predict(test_x)
mse = mean_squared_error(predicts, test_y)

# 可视化
plt.figure(figsize=(10, 7)) # 画布大小
x = np.arange(1, len(predicts) + 1)
plt.plot(x, predicts, label='predict')
plt.plot(x, test_y, label='train_y')
plt.title('the mean_squared_error of LinearRegression is %.2f' % mse)
plt.legend(loc='upper right')

# 对待预测数据进行预测
my_x = pd.read_csv('kc_test.csv', header=None)
my_x.columns = test_columns_name

# 特征缩放
my_minmax_scaler = MinMaxScaler()
my_minmax_scaler.fit(my_x)
my_minmax_scaler = my_minmax_scaler.transform(my_x)
my_scaler_x = pd.DataFrame(my_minmax_scaler, columns=my_x.columns)

my_predicts = lr_model.predict(my_scaler_x)
plt.figure()
plt.plot(range(1, len(my_predicts) + 1), my_predicts)
#plt.show()
result = pd.DataFrame({'SalePrice': my_predicts})

result.to_csv('predict_result', index=False)
