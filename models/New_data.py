import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
import warnings
warnings.filterwarnings("ignore")

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

seed = 7

np.random.seed(seed)
tf.random.set_seed(seed)
df = pd.read_csv('BAU_Data.csv')
df['Date'] = pd.date_range(start = '20210101' , freq = 'H' , periods = len(df))
df.set_index('Date' , inplace = True)
df = df.asfreq('h')
del df['Y']
del df['m']
del df['d']
del df['h']
del df['d-T']
del df['Demand']
df['NetOrder'] = df['Net Order']
df['SPOT Market Volume'] = df['SPOT Market Volume'].shift(24)
df['Market Clearing Price'] = df['Market Clearing Price'].shift(24)
df_data = df
df_data['MONTH'] = [d.strftime('%b') for d in df_data.index]
df_data['HOUR'] = [d.strftime('%H') for d in df_data.index]
df_data['WEEKDAY'] = pd.to_datetime(df_data.index).dayofweek
df_data['WEEKEND'] = 0
df_data.loc[df_data['WEEKDAY'].isin([5, 6]), 'WEEKEND'] = 1  # 5 and 6 weekends
df_data.dropna(inplace = True)
df_data['2_prev'] = df_data['Net Order'].shift(2)
df_data['3_prev'] = df_data['Net Order'].shift(3)
df_data['4_prev'] = df_data['Net Order'].shift(4)
df_data['5_prev'] = df_data['Net Order'].shift(5)
df_data['6_prev'] = df_data['Net Order'].shift(6)
df_data['7_prev'] = df_data['Net Order'].shift(7)
df_data['8_prev'] = df_data['Net Order'].shift(8)
df_data['9_prev'] = df_data['Net Order'].shift(9)
df_data['10_prev'] = df_data['Net Order'].shift(10)
df_data['11_prev'] = df_data['Net Order'].shift(11)
df_data['12_prev'] = df_data['Net Order'].shift(12)
df_data['NetOrder'] = df_data['Net Order']
del df_data['Net Order']
df_data.dropna(inplace = True)
print('Correlations with Net Order: \n')
print(df_data.corrwith(df["NetOrder"]))
print('\n')
print('------------------------------------------------------------------------')
plt.scatter(x='Wind Forecast', y= 'NetOrder', data=df);
plt.ylabel('NetOrder')
plt.xlabel('WindForecast')
plt.legend()
plt.show();
plt.figure(figsize=(25, 12))


hm = sns.heatmap(df_data.corr(), vmin=-1, vmax=1, annot=True)
hm.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
train_size = int(len(df_data) * 0.9)
test_size = len(df_data) - train_size
train, test = df_data.iloc[0:train_size], df_data.iloc[train_size:len(df_data)]
print('\n')
print('Len of train and len of Test: \n')
print(len(train), len(test))
print('--------------------------------------------------')
print('\n')




df_data['MONTH'] = [d.strftime('%m') for d in df_data.index]

from sklearn.preprocessing import RobustScaler

f_columns = ['2_prev', '3_prev', '4_prev','5_prev','6_prev','7_prev','8_prev','9_prev','10_prev','11_prev','12_prev','MONTH','HOUR','WEEKDAY','WEEKEND'
            ,'Demand Forecast' , 'SPOT Market Volume' , 'Wind Forecast',
            'RoR Forecast' , 'Yuk Tahmin Planı (MWh)' , 'Market Clearing Price']

f_transformer = RobustScaler()
cnt_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
cnt_transformer = cnt_transformer.fit(train[['NetOrder']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['NetOrder'] = cnt_transformer.transform(train[['NetOrder']])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['NetOrder'] = cnt_transformer.transform(test[['NetOrder']])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs).astype(np.float32), np.array(ys).astype(np.float32)


time_steps = 4


X_train, y_train = create_dataset(train, train.NetOrder, time_steps)
X_test, y_test = create_dataset(test, test.NetOrder, time_steps)
print('X_trainShape , y_train_shape: \n')
print(X_train.shape, y_train.shape)
print('\n')
print('---------------------------------------------------------------------------')
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
model = keras.Sequential()
model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(
      units=1026,
      input_shape=(X_train.shape[1], X_train.shape[2])
    )
  )
)
model.add(keras.layers.Dropout(rate=0.1))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=8,
    validation_split=0.1,
    shuffle = False
)

print(model.summary())
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();




y_pred = model.predict(X_test)
y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = cnt_transformer.inverse_transform(y_pred)

plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Target_load')
plt.xlabel('Time')
plt.legend()
plt.show();



plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Target_load')
plt.xlabel('Time')
plt.legend()
plt.show();



from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test_inv.flatten(),y_pred_inv.flatten()))




