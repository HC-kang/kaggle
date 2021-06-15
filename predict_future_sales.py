# Ignore the warnings
from typing import cast
import warnings
warnings.filterwarnings('ignore')

import os

import pandas as pd
import numpy as np
from itertools import product 
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import GridSearchCV

# Evaluation metrics
# for regression
from sklearn.metrics import mean_squared_log_error, mean_squared_error,  r2_score, mean_absolute_error
# for classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import SimpleRNN, LSTM, GRU

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb

# 경로설정
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/kaggle/competitive-data-science-predict-future-sales')

# 파일 불러오기
print('--- 파일 불러오기 ---')
item_categories = pd.read_csv('item_categories.csv')
items = pd.read_csv('items.csv')
train = pd.read_csv('sales_train.csv')
shops = pd.read_csv('shops.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')
print('--- 완료 ---')

# #########
# item_categories #(84, 2)
# # 'item_category_name', 'item_category_id']
# items #(22170, 3)
# # 'item_name', 'item_id', 'item_category_id'
# train #(2935849, 6)
# #'date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day'
# shops #(60, 2)
# # shop_name, shop_id
# ###########
# test #(214200, 3)
# # 'ID', 'shop_id', 'item_id'
# sample_submission #(214200, 2)
# # 'ID', 'item_cnt_month'

############
# # 목표 확인..
# sample_submission # 껍데기. ID에 대해 월별 판매량 답안작성
# test # ID는 각 매장별 각 아이템 판매량을 뜻함.

# 데이터 리스트
l = ['item_categories', 'items', 'train', 'shops', 'test', 'sample_submission']
ll = [item_categories, items, train, shops, test, sample_submission]

# # 그냥 다 돌려보기
# for i in range(len(l)):
#     print(f'\n------------{l[i]}.info() ---------------\n', ll[i].info())
#     print(f'\n------------{l[i]}.shape() ---------------\n', ll[i].shape)
#     print(f'\n------------{l[i]}.describe() ---------------\n', ll[i].describe())

# 결측치 확인
print('--- 결측치 확인 ---')
for i in range(len(l)):
    print(f'\n   {l[i]} :\n', ll[i].isna().sum())
print('--- 완료 ---')
# 결측치 없음

# 중복 확인
print('--- 중복 확인 ---')
for i in range(len(l)):
    print(f'\n   {l[i]} :', ll[i].duplicated().sum())
print('--- 완료 ---')

#train 에 중복 6개 식별 -> 삭제
train[train.duplicated()==True]
print('전:',train.shape) #(2935849, 6)
train.drop_duplicates(inplace=True)
print('후:',train.shape) #(2935843, 6) - 6개 감소

# 카테고리 합치기
data = pd.merge(train, items, on = 'item_id', how = 'left')
data.drop('item_name',axis = 1, inplace = True)

# date가 object임. 날짜로 바꿔주려고 해보니, 연월일 순서가 달라서 이상하게 바뀜. 결국 date 전체를 갈아치움
data.info()
data['date'] = data['date'].apply(lambda x: x[-4:]+'-'+x[3:5]+'-'+x[:2])

# 다시 datetime 으로 바꿔주기
data['date'] = pd.to_datetime(data['date'])
data.info()
data

# 다시 한 번 살펴보기
# 근데 데이터에 음수가 있다 : item_cnt_day와 item_price
data.describe()
data['item_cnt_day']
list(data[data['item_cnt_day']<0]['item_cnt_day'].unique())
data[data['item_price']<0] # 1개
data[data['item_cnt_day']<0] # 7356개

# 이리저리 찍어봐도 환불로 인한 것은 아니라고 보임.
# 비슷한 날짜에 개수가 같은것은 커녕, 같은 item_id도 없음.
# data[(data['date']=='2013-01-05')&(data['item_id']==2552)]

# data[data['item_cnt_day']<0]['shop_id'].unique()
# len(data[data['item_cnt_day']<0]['shop_id'].unique())
# data[data['item_cnt_day']<0]['shop_id'].value_counts()
# data[data['item_cnt_day']<0]['shop_id'].value_counts().sum()

# data[data['item_cnt_day']<0]['item_id'].unique()
# len(data[data['item_cnt_day']<0]['item_id'].unique())
# data[data['item_cnt_day']<0]['item_id'].value_counts()
# data[data['item_cnt_day']<0]['item_id'].value_counts().sum()

# data[data['item_cnt_day']==-22]
# data[(data['date']=='2013-11-02')&(data['item_id']==8023)]
# data[(data['item_id']==8023)].tail(50)
# data[(data['date']=='2013-11-02')]

# 결국 음수값들 떨어내기.
data.shape # (2935843, 7)
data = data[data['item_cnt_day']>0]
data.shape # (2928487, 7)

data.shape # (2928487, 7)
data = data[data['item_price']>0]
data.shape # (2928486, 7)

# # 다시 살펴보기.. 가게랑 물건 종류가 다름.
# print('data shop 수:', len(data.shop_id.unique())) # 60
# print('test shop 수:', len(test.shop_id.unique())) # 42
# print('data item 수:', len(data.item_id.unique())) # 21804
# print('test item 수:', len(test.item_id.unique())) # 5100

# # 수만 다른지, 내용물도 다른지 까봐야지
# len(set(data.shop_id.unique()) - set(test.shop_id.unique())) # 18
# len(set(test.shop_id.unique()) - set(data.shop_id.unique())) # 0
#     # shop_id의 경우, 트레인에는 테스트에 없는 것도 있다 + 테스트에는 트레인에 있는 모든 게 있다.
#     # train >>>> test
# len(set(data.item_id.unique()) - set(test.item_id.unique())) #17067
# len(set(test.item_id.unique()) - set(data.item_id.unique())) #363
#     # item_id의 경우, train의 대부분이 test에선 사라졌다
#     # 추가로 test에서 새로 생긴 게 363가지이다.

# # 설마, shops 와 items도 검증.
# len(set(shops.shop_id.unique()) - set(data.shop_id.unique())) # 0
# len(set(data.shop_id.unique()) - set(shops.shop_id.unique())) # 0
#     # 동일함
# len(set(items.item_id.unique()) - set(data.item_id.unique())) # 366
# len(set(data.item_id.unique()) - set(items.item_id.unique())) # 0
# len(set(items.item_id.unique()) - set(test.item_id.unique())) # 17070
# len(set(test.item_id.unique()) - set(items.item_id.unique())) # 0
#     # 다행히 items 가 더 큰 범주로 다 포함함. 제한사항 없음.
#     # 당연히 아이템 카테고리도 전체를 포괄할 수 있음.

# 추가로 이상치 있는지도 확인
# fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# sns.boxplot(data['item_price'], ax=ax[0])
# sns.boxplot(data['item_cnt_day'], ax=ax[1])

data['item_price'].quantile(0.9999) # 29990
data[data['item_price']>50000] # 3개 뿐
# 일단 눈에띄게 떨어져있는 몇개만 제거
print('제거 전 data.shape:', data.shape) #(2928486, 7)
data = data[data['item_price']<250000] 
print('제거 후 data.shape:', data.shape) #(2928485, 7)

data['item_cnt_day'].quantile(0.9999) # 67.15139999985695
data[data['item_cnt_day']>500]
print('제거 전 data.shape:', data.shape) #(2928485, 7)
data = data[data['item_cnt_day']<1000] 
print('제거 후 data.shape:', data.shape) #(2928483, 7)

# 이상치 제거 후 다시 그려보기
# fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# sns.boxplot(data['item_price'], ax=ax[0])
# sns.boxplot(data['item_cnt_day'], ax=ax[1])

# # date_block_num은 뭔지 모르겠음.
# data['date_block_num'].unique() # 0~33까지.
# data['date_block_num'].value_counts()
# data[data['date_block_num']==0]
# data.head(10)

# 정렬된게 날짜순이 아니었음
data = data.sort_values('date')
data = data.reset_index(drop=True)
data

# # 모르겠으니 일단 가게별 상태보기.
# data_shop = data.groupby('shop_id').sum()
# data_shop
# # 합쳐놓고 보니, 건질게 쓸모없음. 월별로 합쳐야함.

data


shops['city2'] = shops['shop_name'].apply(lambda x: x.split()[0])

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

shops['city2'] = encoder.fit_transform(shops['city2'])
shops

item_categories['cat_name'] = item_categories['item_category_name'].apply(lambda x: x.split()[0])
item_categories['splitted'] = item_categories['item_category_name'].apply(lambda x: x.split('-'))

item_categories['cat_name2'] = item_categories['splitted'].apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
item_categories.drop('splitted', axis = 1, inplace= True)
item_categories

encoder = LabelEncoder()
item_categories['cat_name3'] = encoder.fit_transform(item_categories[['cat_name']])
item_categories['cat_name4'] = encoder.fit_transform(item_categories[['cat_name2']])
item_categories

data
#data = pd.merge(data, shops, on = 'shop_id', how = 'left')
# data.drop(['shop_name', 'city2'], axis = 1, inplace=True)
data

item_categories
item_categories.drop(['cat_name', 'cat_name2'], axis =1 , inplace=True)
# data = pd.merge(data, item_categories, on = 'item_category_id', how = 'left')
# data.drop(['item_category_name', 'cat_name', 'cat_name2'], axis = 1, inplace=True)
data

# data.columns = ['date', 'date_block_num', 'shop_id', 'item_id','item_price','item_cnt_day', 'item_category_id','city', 'cat_name','cat_name2']

data


data2 = []

for i in range(34):
    sales = data[data['date_block_num']==i]
    data2.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique()))))

data2

# 쌓기
data2 = pd.DataFrame(np.vstack(data2), columns=['date_block_num', 'shop_id', 'item_id'])
data2.sort_values(by=['date_block_num', 'shop_id', 'item_id'], inplace=True)
data2

data['revenue'] = data['item_price']*data['item_cnt_day']
data

# 
data_item_cnt_month = data.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day':'sum'})
data_item_cnt_month.columns = ['item_cnt_month']
data_item_cnt_month

# data2에 item cnt month 컬럼 합치기
data2 = pd.merge(data2, data_item_cnt_month, on=['date_block_num', 'shop_id', 'item_id'], how = 'left')
data2['item_cnt_month']=data2['item_cnt_month'].fillna(0).clip(0, 20)
data2

test
test['date_block_num']=34
test

data2 = pd.concat([data2, test], ignore_index=True, sort=False, keys=['date_block_num', 'shop_id', 'item_id'])
data2.fillna(0, inplace=True)
data2

shops.drop(['shop_name'], axis=1, inplace=True)
shops.columns = ['shop_id', 'city']
shops

item_categories.drop(['item_category_name'], axis = 1, inplace=True)
item_categories
# data
# item_categories=data[['item_id', 'cat_name', 'cat_name2','item_category_id']]
# item_categories.columns=['item_category_id', 'cat_name', 'cat_name2']
# item_categories

items.drop('item_name', axis = 1, inplace=True)
items

data2 = pd.merge(data2, shops, on='shop_id', how='left')
data2
data2 = pd.merge(data2, items, on='item_id', how='left')
data2
data2 = pd.merge(data2, item_categories, on='item_category_id', how='left')
data2.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'ID', 'city','item_category_id', 'cat_name', 'cat_name2']
data2

def lagger(df, lags, col):
    tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
    for i in lags:
        shifted = tmp[:]
        shifted.columns = ['date_block_num', 'shop_id', 'item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return df

data2 = lagger(data2, [1,2,3,6,12], 'item_cnt_month')


# 다이어트
data2.info()
data2['date_block_num']=data2['date_block_num'].astype('int8')
data2['shop_id']=data2['shop_id'].astype('int8')

lll=[1,2,3,6,12]
for i in lll:
    data2[f'item_cnt_month_lag_{i}']=data2[f'item_cnt_month_lag_{i}'].astype('float16')
data2.info()

lll = ['item_category_id', 'cat_name', 'cat_name2']
for i in lll:
    data2[i]=data2[i].astype('int8')
data2.info()
data2['ID'] = data2['ID'].astype('int32')

data2.info()

# 단순 월별 평균 판매량
data_blocknum_cnt_month_mean = data2.groupby(['date_block_num']).agg({'item_cnt_month':'mean'})
data_blocknum_cnt_month_mean.reset_index(inplace=True)
data_blocknum_cnt_month_mean

data2 = pd.merge(data2, data_blocknum_cnt_month_mean, on='date_block_num', how='left')
data2.info()
data2.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'ID','city', 'item_category_id', 'cat_name', 'cat_name2','item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3','item_cnt_month_lag_6', 'item_cnt_month_lag_12', 'item_cnt_month_mean']
data2

# 월, 상점별 평균 판매량
data_shop_cnt_month_mean = data2.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month':'mean'})
data_shop_cnt_month_mean.reset_index(inplace=True)
data_shop_cnt_month_mean.columns=['date_block_num', 'shop_id', 'item_cnt_month_shop_mean']

data2 = pd.merge(data2, data_shop_cnt_month_mean, on=['date_block_num', 'shop_id'], how='left')
data2

# 월, 품목별 평균 판매량
data_item_cnt_month_mean=data2.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month':'mean'})
data_item_cnt_month_mean.reset_index(inplace=True)
data_item_cnt_month_mean.columns=['date_block_num', 'item_id', 'item_cnt_month_item_mean']

data2 = pd.merge(data2, data_item_cnt_month_mean, on=['date_block_num', 'item_id'], how='left')
data2.columns=['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'ID', 'city',
       'item_category_id', 'cat_name', 'cat_name2', 'item_cnt_month_lag_1',
       'item_cnt_month_lag_2', 'item_cnt_month_lag_3', 'item_cnt_month_lag_6',
       'item_cnt_month_lag_12', 'item_cnt_month_mean',
       'item_cnt_month_shop_mean', 'item_cnt_month_item_mean']
data2

# 월 카테고리별 평균 판매량
data_cat_cnt_month_mean = data2.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month':'mean'})
data_cat_cnt_month_mean.reset_index(inplace=True)
data_cat_cnt_month_mean.columns=['date_block_num', 'item_category_id', 'item_cnt_month_cat_mean']
data_cat_cnt_month_mean

data2 = pd.merge(data2, data_cat_cnt_month_mean, on=['date_block_num', 'item_category_id'], how= 'left')
data2.info()

# 월, 가게, 카테고리별 평균 판매량
data_shop_cat_cnt_month_mean = data2.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month':'mean'})
data_shop_cat_cnt_month_mean.reset_index(inplace=True)
data_shop_cat_cnt_month_mean.columns=['date_block_num', 'shop_id', 'item_category_id', 'item_cnt_month_shop_cat_mean']

data2 = pd.merge(data2, data_shop_cat_cnt_month_mean, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
data2

# 월, 가게, 캣네임1별 평균 판매량
data_shop_catname_cnt_month_mean = data2.groupby(['date_block_num', 'shop_id', 'cat_name']).agg({'item_cnt_month':'mean'})
data_shop_catname_cnt_month_mean.reset_index(inplace=True)
data_shop_catname_cnt_month_mean.columns=['date_block_num', 'shop_id', 'cat_name', 'item_cnt_month_shop_catname_mean']
data_shop_catname_cnt_month_mean

data2 = pd.merge(data2, data_shop_catname_cnt_month_mean, on=['date_block_num', 'shop_id', 'cat_name'], how='left')
data2.info()

# 월, 가게, 캣네임2별 평균 판매량
data_shop_catname2_cnt_month_mean = data2.groupby(['date_block_num', 'shop_id', 'cat_name2']).agg({'item_cnt_month':'mean'})
data_shop_catname2_cnt_month_mean.reset_index(inplace=True)
data_shop_catname2_cnt_month_mean.columns=['date_block_num', 'shop_id', 'cat_name2', 'item_cnt_month_shop_catname2_mean']
data_shop_catname2_cnt_month_mean

data2 = pd.merge(data2, data_shop_catname2_cnt_month_mean, on=['date_block_num', 'shop_id', 'cat_name2'], how='left')
data2.info()

# 월, 도시별 평균 판매량
data_city_cnt_month_mean = data2.groupby(['date_block_num', 'city']).agg({'item_cnt_month':'mean'})
data_city_cnt_month_mean.reset_index(inplace=True)
data_city_cnt_month_mean.columns=['date_block_num', 'city', 'item_cnt_month_city']
data_city_cnt_month_mean

data2 = pd.merge(data2, data_city_cnt_month_mean, on=['date_block_num', 'city'], how = 'left')
data2

# 월, 아이템, 도시별 평균 판매량
data_item_city_cnt_month_mean = data2.groupby(['date_block_num', 'item_id', 'city']).agg({'item_cnt_month':'mean'})
data_item_city_cnt_month_mean.reset_index(inplace=True)
data_item_city_cnt_month_mean.columns=['date_block_num', 'item_id', 'city', 'item_cnt_month_item_city']
data_item_city_cnt_month_mean

data2 = pd.merge(data2, data_item_city_cnt_month_mean, on=['date_block_num', 'item_id', 'city'], how='left')
data2

# 월, 캣네임 별 평균 판매량 
data_catname_cnt_month_mean = data2.groupby(['date_block_num', 'cat_name']).agg({'item_cnt_month':'mean'})
data_catname_cnt_month_mean.reset_index(inplace=True)
data_catname_cnt_month_mean.columns=['date_block_num', 'cat_name', 'item_cnt_month_catname']
data_catname_cnt_month_mean

data2 = pd.merge(data2, data_catname_cnt_month_mean, on=['date_block_num', 'cat_name'], how = 'left')
data2.columns


# 월, 캣네임2 별 평균 판매량 
data_catname2_cnt_month_mean = data2.groupby(['date_block_num', 'cat_name2']).agg({'item_cnt_month':'mean'})
data_catname2_cnt_month_mean.reset_index(inplace=True)
data_catname2_cnt_month_mean.columns=['date_block_num', 'cat_name2', 'item_cnt_month_catname2']
data_catname2_cnt_month_mean

data2 = pd.merge(data2, data_catname2_cnt_month_mean, on=['date_block_num', 'cat_name2'], how = 'left')
data2


# 2차 다이어트
lll=['item_cnt_month_mean',
       'item_cnt_month_shop_mean', 'item_cnt_month_item_mean',
       'item_cnt_month_cat_mean', 'item_cnt_month_shop_cat_mean',
       'item_cnt_month_shop_catname_mean', 'item_cnt_month_shop_catname2_mean',
       'item_cnt_month_city', 'item_cnt_month_item_city',
       'item_cnt_month_catname','item_cnt_month_catname2']
for i in lll:
    data2[i]=data2[i].astype('float16')
data2.info()
data2
data

data_item_price_mean = data.groupby('item_id').agg({'item_price':'mean'})
data_item_price_mean.reset_index(inplace=True)
data_item_price_mean.columns = ['item_id', 'item_price_mean']
data_item_price_mean

data2=pd.merge(data2, data_item_price_mean, on=['item_id'], how='left')
data2

data_item_price_month_mean = data.groupby(['date_block_num', 'item_id']).agg({'item_price':'mean'})
data_item_price_month_mean.reset_index(inplace=True)
data_item_price_month_mean.columns = ['date_block_num', 'item_id', 'item_price_month_mean']
data_item_price_month_mean

data2 = pd.merge(data2, data_item_price_month_mean, on=['date_block_num', 'item_id'], how = 'left')
data2

data2.info()
data2['item_price_mean'] = data2['item_price_mean'].astype('float16')
data2['item_price_month_mean'] = data2['item_price_month_mean'].astype('float16')
data2.info()

data2 = lagger(data2, [1,2,3,4,5,6], 'item_price_month_mean')
data2.columns

for i in range(1, 6+1):
    data2['price_lag_'+str(i)] = (data2['item_price_month_mean_lag_'+str(i)]-data2['item_price_mean']) / data2['item_price_mean']

data2.info()
data2

def select_trend(row):
    for i in range(1, 6+1):
        if row['price_lag_'+str(i)]:
            return row['price_lag_'+str(i)]
    return 0

data2['price_lag'] = data2.apply(select_trend, axis = 1)
data2['price_lag'] = data2['price_lag'].astype('float16')
data2['price_lag']
data2.info()

data2.columns
#TODO:
# data2.drop(['item_price_mean',
#        'item_price_month_mean', 'item_price_month_mean_lag_1',
#        'item_price_month_mean_lag_2', 'item_price_month_mean_lag_3',
#        'item_price_month_mean_lag_4', 'item_price_month_mean_lag_5',
#        'item_price_month_mean_lag_6', 'price_lag_1', 'price_lag_2',
#        'price_lag_3', 'price_lag_4', 'price_lag_5', 'price_lag_6'], axis = 1, inplace=True)

data2

data_shop_month_revenue_sum = data.groupby(['date_block_num', 'shop_id']).agg({'revenue':'sum'})
data_shop_month_revenue_sum.reset_index(inplace=True)
data_shop_month_revenue_sum.columns = ['date_block_num', 'shop_id', 'shop_month_revenue']
data_shop_month_revenue_sum

data2 = pd.merge(data2,data_shop_month_revenue_sum, on=['date_block_num', 'shop_id'], how = 'left')
data2

data_shop_month_revenue_sum = data_shop_month_revenue_sum.groupby('shop_id').agg({'shop_month_revenue':'mean'})
data_shop_month_revenue_sum.columns = ['shop_month_revenue_mean']
data_shop_month_revenue_sum.reset_index(inplace=True)
data_shop_month_revenue_sum

data2=pd.merge(data2, data_shop_month_revenue_sum, on=['shop_id'], how = 'left')
data2

data2['delta_revenue'] = (data2['shop_month_revenue'] - data2['shop_month_revenue_mean'])/data2['shop_month_revenue_mean']

data2 = lagger(data2, [1], 'delta_revenue')
data2.columns

data2.drop(['shop_month_revenue', 'shop_month_revenue_mean', 'delta_revenue'], axis = 1, inplace=True)
data2

data2['month'] = data2['date_block_num'] % 12

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

data2['days'] = data2['month'].map(days)
data2
data2.info()
data2['days']=data2['days'].astype('int8')
data2['delta_revenue_lag_1']=data2['delta_revenue_lag_1'].astype('float16')
data2['item_cnt_month']=data2['item_cnt_month'].astype('float16')
data2.info()

cache = {}
data2['item_shop_last_sale'] = -1
data2['item_shop_last_sale'] = data2['item_shop_last_sale'].astype('int8')
for idx, row in tqdm(data2.iterrows()):
    key = str(row.item_id)+' '+str(row.shop_id)
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        data2.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
        cache[key] = row.date_block_num


cache = {}
data2['item_last_sale'] = -1
data2['item_last_sale'] = data2['item_last_sale'].astype('int8')
for idx, row in tqdm(data2.iterrows()):
    key = row.item_id
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        if row.date_block_num > last_date_block_num:
            data2.at[idx, 'item_last_sale']=row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num

data2['item_shop_first_sale'] = data2['date_block_num'] - data2.groupby(['item_id', 'shop_id'])['date_block_num'].transform('min')
data2['item_first_sale'] = data2['date_block_num'] - data2.groupby('item_id')['date_block_num'].transform('min')


data2.info()
data2





import gc
data2.to_pickle('.pkl')
del data2
del cache
del items
del shops
del train
del data
gc.collect();












# data = pd.read_pickle('data2.pkl')
data = pd.read_pickle('full_data.pkl')
# data.to_csv('full_data.csv')
data.columns
data.isna().sum()
data
# ~12월 드롭
data = data[data['date_block_num']>11]

# lag 결측치 평균값으로 채우기
def fill_mean(df):
    for col in df.columns:
        tmp=df[col].mean()
        if ('_lag_' in col):
            df[col].fillna(tmp, inplace=True)
    return df
data.describe()
len(data.columns)


# 결측치 0으로 채우기
def fill_na(df):
    for col in df.columns:
        df[col].fillna(0, inplace=True)
    return df
data.info()
# data = fill_mean(data)
data = fill_na(data)

data.item_first_sale.describe()
data.item_first_sale.describe()

data.drop(
    ['item_last_sale',
     'price_lag',
     'price_lag_1',
     'item_cnt_month_mean',
     'item_cnt_month_lag_2',
     'item_price_month_mean_lag_1',
     
     ], axis = 1, inplace=True)



data = data[['shop_id', 'ID','date_block_num',
       'item_cnt_month_lag_1', 'item_cnt_month_shop_mean',
       'item_cnt_month_item_mean', 'item_cnt_month_cat_mean',
       'item_cnt_month_shop_cat_mean', 'item_cnt_month_shop_catname_mean',
       'item_cnt_month_shop_catname2_mean', 'item_cnt_month_city',
       'item_cnt_month_item_city',
       'item_cnt_month_catname2', 'item_cnt_month']]

data.columns
data = data[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'ID', 'city',
       'item_category_id', 'cat_name', 'cat_name2', 'item_cnt_month_lag_1',
       'item_cnt_month_lag_2', 'item_cnt_month_lag_3',
       'item_cnt_month_mean',
       'item_cnt_month_shop_mean', 'item_cnt_month_item_mean',
       'item_cnt_month_cat_mean', 'item_cnt_month_shop_cat_mean',
       'item_cnt_month_shop_catname_mean', 'item_cnt_month_shop_catname2_mean',
       'item_cnt_month_city', 'item_cnt_month_item_city',
       'item_cnt_month_catname', 'item_cnt_month_catname2', 'item_price_mean',
       'item_price_month_mean', 'item_price_month_mean_lag_1',
       'price_lag_1', 'price_lag_2',
       'delta_revenue_lag_1',
       'item_shop_first_sale','item_first_sale']]

data = data[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month',
       'item_category_id','cat_name2',
       'item_cnt_month_mean',
       'item_cnt_month_shop_mean', 'item_cnt_month_item_mean',
       'item_cnt_month_cat_mean', 'item_cnt_month_shop_cat_mean',
       'item_cnt_month_shop_catname_mean', 'item_cnt_month_shop_catname2_mean',
       'item_cnt_month_city', 'item_cnt_month_item_city',
       'item_cnt_month_catname', 'item_cnt_month_catname2', 'item_price_mean',
       'item_price_month_mean',
       'item_shop_first_sale', 'item_first_sale']]


data = data[['date_block_num', 'shop_id',
       'item_cnt_month_shop_mean','item_cnt_month_item_city','item_cnt_month']]


# 없던 363개에 initial sale 추가하기
data['item_first_sale'] = data.groupby(['item_id'])['date_block_num'].transform('min')
data['item_shop_first_sale'] = data.groupby(['item_id', 'shop_id'])['date_block_num'].transform('min')
data['item_first_sale'] = data2['date_block_num'] - data2.groupby('item_id')['date_block_num'].transform('min')
data[data['date_block_num']==34]['item_shop_first_sale']
# 신상품 363개 목록
new_item = list(set(test.item_id.unique()) - set(train.item_id.unique()))
new_item

data[data['item_id']==8182]
data.item_first_sale.mean()
data['item_fitst_sale2'] = data['item_first_sale'].mean() if 

data
#data.drop(['item_last_sale', 'item_cnt_month_lag_12', 'days', 'cat_name','item_cnt_month_lag_6'], axis = 1, inplace=True)


X_train = data[data['date_block_num']<33].drop('item_cnt_month', axis = 1)
y_train = data[data['date_block_num']<33]['item_cnt_month']
X_valid = data[data['date_block_num']==33].drop('item_cnt_month', axis = 1)
y_valid = data[data['date_block_num']==33]['item_cnt_month']
X_test = data[data['date_block_num']==34].drop('item_cnt_month', axis = 1)
X_train

X_train.drop('date_block_num', axis = 1, inplace=True)
X_valid.drop('date_block_num', axis = 1, inplace=True)
X_test.drop('date_block_num', axis = 1, inplace=True)

del data
def X_train
def y_train
def X_valid
def y_valid
def x_test
gc.collect();


# 티스토리 참고
from xgboost import XGBRegressor
model = XGBRegressor(
    max_depth = 8,
    n_estimators = 1000,
    min_child_weight=300,
    colsample_bytree = 0.8,
    subsample=0.8,
    eta=0.3,
    seed=42)

# 랜덤서치 결과
model = XGBRegressor(
    max_depth = 10,
    n_estimators = 8,
    learning_rate = 0.1,
    seed=42)

# 훔치기!!!
model = XGBRegressor(
    max_depth = 11, 
    min_child_weight=0.5, 
    subsample = 1, 
    eta = 0.3, 
    num_round = 1000, 
    seed = 1, 
    nthread = 16
)


model.fit(
    X_train,
    y_train,
    eval_metric='rmse',
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True,
    early_stopping_rounds=10
)

y_pred = model.predict(X_valid).clip(0, 20)
y_test_xgboost = model.predict(X_test).clip(0, 20)

# 기본모델
xgb = XGBClassifier(tree_method='hist', random_state = 42)
scores = cross_validate(xgb, train_input, train_target, return_train_score = True, n_jobs = -1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

from lightgbm import LGBMRegressor
lgbm = LGBMRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=42
)
lgbm.fit(
    X_train,
    y_train,
    eval_metric='rmse',
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True,
    early_stopping_rounds=30
)

y_test_lgbm = lgbm.predict(X_test).clip(0, 20)


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(
    n_estimators=25,
    random_state=42, 
    max_depth=15, 
    n_jobs=-1
)
rf.fit(
    X_train,
    y_train,
)

y_test_rf = rf.predict(X_test).clip(0, 20)






#--*-*-*-*-제출셋 만들기
test
submission = pd.DataFrame(
    {
        'ID':test.index,
        'item_cnt_month':y_test_rf,
    }
)

submission['item_cnt_month'] = (submission['item_cnt_month1']+submission['item_cnt_month2']+submission['item_cnt_month3'])/3

submission.drop(['item_cnt_month1', 'item_cnt_month2','item_cnt_month3'], axis = 1, inplace=True)

submission

submission.to_csv('rf_submission_3.csv', index=False)

from xgboost import plot_importance
fig, ax = plt.subplots(1, 1, figsize = (10, 14))
plot_importance(model, ax = ax)

model.save_model('.model')


from lightgbm import plot_importance
fig, ax = plt.subplots(1, 1, figsize = (10, 14))
plot_importance(lgbm, ax = ax)






params = {
    'n_estimators': [8, 16, 24],
    'max_depth': [3, 5, 7, 10],
    'num_iterations': [1000, 1500],
    'learning_rate': [0.05, 0.07, 0.1],
    'colsample_bytree': [0.5, 1],
    'subsample': [0.7, 0.8],
}

lgbm = LGBMRegressor(
    n_estimators=params['n_estimators'],
    max_depth=params['max_depth'],
    num_iterations=params['num_iterations'],
    learning_rate=params['learning_rate'],
    colsample_bytree=params['colsample_bytree'],
    subsample=params['subsample'],
)

xgboost = XGBRegressor(
    n_estimators=params['n_estimators'],
    max_depth=params['max_depth'],
    num_iterations=params['num_iterations'],
    learning_rate=params['learning_rate'],
    colsample_bytree=params['colsample_bytree'],
    subsample=params['subsample'],
)

models = [lgbm, xgboost]

# 두 모델의 fit, score을 포문으로 돌림
for i in models:
    grid_model = GridSearchCV(i, param_grid=params,
                              scoring='neg_mean_squared_error',
                              cv=5,
                              n_jobs=5,
                              )

    grid_model.fit(X_train, y_train,
                   eval_metric='rmse',
                   eval_set=[(X_train, y_train), (X_valid, y_valid)],
                   verbose=True,
                   early_stopping_rounds=10
                   )

    y_pred = i.predict(X_valid).clip(0, 20)
    y_test = i.predict(X_test).clip(0, 20)

    # gridsearch의 결과들은
    # grid_model.cv_results_
    # 로 들어간다.

    # 파라미터 조합은 params에 들어가있고 점수는 mean_test_score에 들어가있다.
    params= grid_model.cv_results_['params']
    score = grid_model.cv_results_['neg_mean_test_score']

    # 파라미터 조합과 점수를 데이터프레임으로 만들자!!
    results = pd.DataFrame(params)

    # 이 score를 RMSE로 바꿔주려면 양수로 만들고 루트를 씌워주어야 한다.
    results['RMSE'] = np.sqrt(-1 * results['score'])
    results['score']= score
    print('-----score-----')
    print(results['score'])

    # submission 만들기############
    submission = pd.DataFrame(
        {
            'ID': test.index,
            'item_cnt_month': y_test
        }
    )
    submission.to_csv('{}_submission.csv'.format(i), index=False)

    # 그래프 그리기
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    plot_importance(i, ax=ax)

    # 모델 저장하기
    i.save_model('first_{}.model'.format(i))





##############


import lightgbm as lgb
feature_name = X_train.columns.tolist()

params = {
    'objective': 'mse',
    'metric': 'rmse',
    'num_leaves': 2 ** 8 -1,
    'learning_rate': 0.005,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'seed': 1,
    'verbose': 1
}
feature_name_indexes = [ 
                        'city', 
                        'item_category_id',
                        'cat_name', 
                        'shop_id',
]

lgb_train = lgb.Dataset(X_train[feature_name], y_train)
lgb_eval = lgb.Dataset(X_valid[feature_name], y_valid, reference=lgb_train)

evals_result = {}
gbm = lgb.train(
        params, 
        lgb_train,
        num_boost_round=3000,
        valid_sets=(lgb_train, lgb_eval), 
        feature_name = feature_name,
        categorical_feature = feature_name_indexes,
        verbose_eval=50, 
        evals_result = evals_result,
        early_stopping_rounds = 10)




Y_test = gbm.predict(X_test[feature_name]).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('gbm_submission.csv', index=False)










# # 월별로 상점과 아이템에 대해서 묶어봐야겠음.
# # year_month 컬럼 만들어주기
# data['year_month'] = data['date'].dt.strftime('%Y-%m')
# data

# # date_block_num이 월별로 묶인거였음.. 버림.
# data.drop('date_block_num', axis = 1, inplace = True)
# data

data

X = data.drop('date', axis = 1).groupby('date_block_num').mean()
X = data.drop('date', axis = 1).groupby('date_block_num').sum()
X

# 그룹핑 해주기
# 월 단위로 합쳐보니 건질것은 월 판매량뿐..
data_shop = data.groupby(['year_month', 'shop_id']).sum()
data_shop_cnt_month = data_shop[['item_cnt_day']]
data_shop_cnt_month.columns = ['shop_cnt_month']
data_shop_cnt_month # (1582, 1)

# 상품별도 역시 월 판매량
data_item = data.groupby(['year_month', 'item_id']).sum()
data_item_cnt_month = data_item[['item_cnt_day']]
data_item_cnt_month.columns = ['item_cnt_month']
data_item_cnt_month #(233802, 1) 그래프 그릴 엄두가 안남...

# 카테고리별도 월 판매량 종합해봄.
data_item_cat = data.groupby(['year_month', 'item_category_id']).sum()
data_item_cat_cnt_month = data_item_cat[['item_cnt_day']]
data_item_cat_cnt_month.columns = ['item_cat_cnt_month']
data_item_cat_cnt_month #(2076, 1) 비벼볼만 해 보임.

# 0번 상점, 상품, 카테고리 수 확인
data_shop_cnt_month[data_shop_cnt_month.index.get_level_values('shop_id')==0]
data_item_cnt_month[data_item_cnt_month.index.get_level_values('item_id')==0]
data_item_cat_cnt_month[data_item_cat_cnt_month.index.get_level_values('item_category_id')==0]


# 그룹핑 추가 - 상점, 아이템별
data_shop_item_sum = data.groupby(['year_month', 'shop_id', 'item_id']).sum()
data_shop_item_mean = data.groupby(['year_month', 'shop_id', 'item_id']).mean()
data_shop_item_sum
data_shop_item_mean
data_shop_item_mean.columns

X=data_shop_item_mean[['item_price','item_category_id']]
y=data_shop_item_sum['item_cnt_day']
X.shape
y
####

# 스플릿
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)

print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)


#스케일링
## Scaling
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled
X_test_scaled

# 걍 꼬라박1
from xgboost import XGBRegressor
model_reg = XGBRegressor()
model_reg.fit(X_train_scaled, y_train)

# 꼬라박1
y_pred = model_reg.predict(X_test_scaled)

plt.figure(figsize = (10, 10))
plt.scatter(x = y_test, y = y_pred)
plt.show()

test

submission = sample_submission[:]
submission




############
RF_reg = RandomForestRegressor(n_estimators = 100)
RF_reg.fit(x_train, y_train)

val_pred = RF_reg.predict(x_val)
train_pred = RF_reg.predict(x_train)
print(val_pred)
print(train_pred)
print_evaluate(y_val, val_pred)
print_evaluate(y_train, train_pred)


def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')







####

# 같은 가게에서, 같은 제품을 다른 가격에 파는 경우도 있음..
data[(data['year_month']=='2013-01') & (data['shop_id']==0) & (data['item_id']==51)]

# 상점별 매출
data_shop_cnt_month.unstack(level=1)
plt.figure(figsize = (10, 10))
plt.plot(data_shop_cnt_month.unstack(level=1).index, data_shop_cnt_month.unstack(level=1)['shop_cnt_month'])
plt.xticks(rotation = 90)
plt.show() # 11, 12월에 상점 매출 오름

# 상품별 매출
data_item_cnt_month.unstack(level=1)
plt.figure(figsize = (10, 10))
plt.plot(data_item_cnt_month.unstack(level=1).index, data_item_cnt_month.unstack(level=1)['item_cnt_month'])
plt.xticks(rotation = 90)
plt.show() # 출시 직후 높다가 점점 떨어짐. 12000 넘는 이상치? 도 하나 있는데 카테고리 확인해야할듯.

data_item_cnt_month[data_item_cnt_month['item_cnt_month']>12000] # 20949번 item
# Фирменный пакет майка 1С Интерес белый
# 카테고리 71

# 카테고리별 매출
data_item_cat_cnt_month.unstack(level=1)
plt.figure(figsize = (10, 10))
plt.plot(data_item_cat_cnt_month.unstack(level=1).index, data_item_cat_cnt_month.unstack(level=1)['item_cat_cnt_month'])
plt.xticks(rotation = 90)
plt.show() # 11, 12월에 상점 매출 오름



# 시계열성 배재하고 돌려보기
data_nodate = data.drop(['date', 'year_month'], axis=1)
data_nodate.info()

len(data.columns)
data.corr()
test[test['item_id']==20949].count()
data_nodate.columns
data_nodate
data


# 모르겠고 일단 그래프 그려보기
fig, ax = plt.subplots(3, 2, figsize=(20, 20))
count = 0
columns = data_nodate.columns
for row in tqdm(range(3)):
    for col in range(2):
        sns.kdeplot(data_nodate[columns[count]], ax=ax[row][col])
        ax[row][col].set_title(columns[count], fontsize=15)
        count+=1
        if count == 5 :
            break

# 로그변환
skew_columns = ['item_price','item_cnt_day']
for c in skew_columns:
    data_nodate[c] = np.log1p(data_nodate[c].values)

# 다시 그래프 그려보기
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.kdeplot(data_nodate['item_price'], ax=ax[0])
sns.kdeplot(data_nodate['item_cnt_day'], ax=ax[1])


# 나눌 준비
X = data_nodate
y = X['item_cnt_day']
del X['item_cnt_day']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)

print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)


gboost = GradientBoostingRegressor(random_state=1)
xgboost = xgb.XGBRegressor(random_state=1)
lightgbm = lgb.LGBMRegressor(random_state=1)

models = [
    {'model':gboost, 'name':'GradientBoosting'}, {'model':xgboost, 'name':'XGBoost'},
    {'model':lightgbm, 'name':'LightGBM'}
]


def get_cv_score(models):                        #TODO:
    kfold = KFold(n_splits=5, random_state=1, shuffle=True).get_n_splits(X_train.values)
    for m in models:
        print("Model {} CV score : {:.4f}".format(m['name'], np.mean(cross_val_score(m['model'], X_train.values, y_train)), 
                                             kf=kfold))


get_cv_score(models)

