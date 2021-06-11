# Ignore the warnings
import warnings
warnings.filterwarnings('ignore')

# System related and data input controls
import os

# Data manipulation, visualization and useful functions
import pandas as pd
import numpy as np
from itertools import product # iterative combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling algorithms
# General(Statistics/Econometrics)
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Model selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Evaluation metrics
# for regression
from sklearn.metrics import mean_squared_log_error, mean_squared_error,  r2_score, mean_absolute_error
# for classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import SimpleRNN, LSTM, GRU

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

#########
item_categories #(84, 2)
# 'item_category_name', 'item_category_id']
items #(22170, 3)
# 'item_name', 'item_id', 'item_category_id'
train #(2935849, 6)
#'date', 'date_block_num', 'shop_id', 'item_id', 'item_price', TODO:'item_cnt_day'
shops #(60, 2)
# shop_name, shop_id
###########
test #(214200, 3)
# 'ID', 'shop_id', 'item_id'
sample_submission #(214200, 2)
# 'ID', TODO:'item_cnt_month'

############
# 데이터 리스트
l = ['item_categories', 'items', 'train', 'shops', 'test', 'sample_submission']
ll = [item_categories, items, train, shops, test, sample_submission]

# 그냥 다 돌려보기
for i in range(len(l)):
    print(f'\n------------{l[i]}.info() ---------------\n', ll[i].info())
    print(f'\n------------{l[i]}.shape() ---------------\n', ll[i].shape)
    print(f'\n------------{l[i]}.describe() ---------------\n', ll[i].describe())

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

# 그룹별로 모아보려다가, 일단 패스
train
train_item_group = train.groupby('item_id').sum()
train_item_group

# 일단 최대한 필요해보이는 것들 합쳐보자. 카테고리 합치기
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
data[(data['date']=='2013-01-05')&(data['item_id']==2552)]

data[data['item_cnt_day']<0]['shop_id'].unique()
len(data[data['item_cnt_day']<0]['shop_id'].unique())
data[data['item_cnt_day']<0]['shop_id'].value_counts()
data[data['item_cnt_day']<0]['shop_id'].value_counts().sum()

data[data['item_cnt_day']<0]['item_id'].unique()
len(data[data['item_cnt_day']<0]['item_id'].unique())
data[data['item_cnt_day']<0]['item_id'].value_counts()
data[data['item_cnt_day']<0]['item_id'].value_counts().sum()

data[data['item_cnt_day']==-22]
data[(data['date']=='2013-11-02')&(data['item_id']==8023)]
data[(data['item_id']==8023)].tail(50)
data[(data['date']=='2013-11-02')]

# 결국 음수값들 떨어내기.
data.shape # (2935843, 7)
data = data[data['item_cnt_day']>0]
data.shape # (2928487, 7)

# 다시 살펴보기.. 가게랑 물건 종류가 다름.
print('data shop 수:', len(data.shop_id.unique())) # 60
print('test shop 수:', len(test.shop_id.unique())) # 42
print('data item 수:', len(data.item_id.unique())) # 21804
print('test item 수:', len(test.item_id.unique())) # 5100

# 수만 다른지, 내용물도 다른지 까봐야지
len(set(data.shop_id.unique()) - set(test.shop_id.unique())) # 18
len(set(test.shop_id.unique()) - set(data.shop_id.unique())) # 0
    # shop_id의 경우, 트레인에는 테스트에 없는 것도 있다 + 테스트에는 트레인에 있는 모든 게 있다.
    # train >>>> test
len(set(data.item_id.unique()) - set(test.item_id.unique())) #17067
len(set(test.item_id.unique()) - set(data.item_id.unique())) #363
    # item_id의 경우, train의 대부분이 test에선 사라졌다
    # 추가로 test에서 새로 생긴 게 363가지이다.

# 설마, shops 와 items도 검증.
len(set(shops.shop_id.unique()) - set(data.shop_id.unique())) # 0
len(set(data.shop_id.unique()) - set(shops.shop_id.unique())) # 0
    # 동일함
len(set(items.item_id.unique()) - set(data.item_id.unique())) # 366
len(set(data.item_id.unique()) - set(items.item_id.unique())) # 0
len(set(items.item_id.unique()) - set(test.item_id.unique())) # 17070
len(set(test.item_id.unique()) - set(items.item_id.unique())) # 0
    # 다행히 items 가 더 큰 범주로 다 포함함. 제한사항 없음.
    # 당연히 아이템 카테고리도 전체를 포괄할 수 있음.

# date_block_num은 뭔지 모르겠음.
data['date_block_num'].unique() # 0~33까지.
data['date_block_num'].value_counts()
data[data['date_block_num']==0]
data.head(10)
# 정렬된게 날짜순이 아니었네...
data = data.sort_values('date')
data = data.reset_index(drop=True)

data['date'].dt.year+data['date'].dt.month
data

# 모르겠으니 일단 가게별 상태보기.
data_shop = data.groupby('shop_id').sum()
data_shop
# 합쳐놓고 보니, 건질게 item_cnt_day뿐.

# 목표 재확인..
sample_submission # 껍데기. ID에 대해 월별 판매량 답안작성
test # ID는 각 매장별 각 아이템 판매량을 뜻함.

# 월별로 상점과 아이템에 대해서 묶어봐야겠음.
# year_month 컬럼 만들어주기
data['year_month'] = data['date'].dt.strftime('%Y-%m')

# 그룹핑 해주기
data_shop = data.groupby(['year_month', 'shop_id']).mean()
data_shop = data.groupby(['year_month', 'shop_id']).sum()
data_shop[data_shop['date_block_num']==1]

# date_block_num이 월별로 묶인거였음.. 버림.
data.drop('date_block_num', axis = 1, inplace = True)

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

plt.plot(data_shop_cnt_month)
plt.scatter(data = data_shop_cnt_month, x = data_shop_cnt_month.index, y = 'shop_cnt_month')


# 일단 이상치 있는지도 확인
sns.boxplot(data['item_price'])
sns.boxplot(data['item_cnt_day'])

data['item_price'].quantile(0.999)
data[data['item_price']>100000]
data[data['item_price']>50000]

data['item_cnt_day'].quantile(0.99)
data[data['item_cnt_day']>1000]
data[data['item_cnt_day']>500]



data
data.corr()

test
sample_submission



# 독립변인 X와 종속변인 y로 나누어주기
data.columns
# X = data[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_category_id']]
X = data[['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_category_id']]
y = data['item_cnt_day']



















# 일단 스케일이고 뭐고 없이 정면돌파
# 트레인/테스트셋 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 형태확인
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

# 걍 꼬라박
from xgboost import XGBRegressor
model_reg = XGBRegressor()
model_reg.fit(X_train, y_train)

from lightgbm import LGBMRegressor
model_lgbm = LGBMRegressor()
model_lgbm.fit(X_train, y_train)

y_pred = model_reg.predict(X_train)
y_pred = model_lgbm.predict(X_test)

plt.scatter(x = y_test, y = y_pred)







data

data2 = data[:]
data2






