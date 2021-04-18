
import numpy as np
import pandas as pd
import seaborn as sns



import matplotlib.pyplot as plt
from matplotlib import rc
from pylab import rcParams
from datetime import datetime
import matplotlib.pyplot as plt 
!pip install --quiet pytorch-lightning==1.2.5
#progress bars and much more
#TODO: read documentation
!pip install --quiet tqdm==4.59.0

from tqdm.notebook import tqdm
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

rcParams['figure.figsize'] = 13,7

tqdm.pandas()

pl.seed_everything(7)

df = pd.read_csv('/content/drive/MyDrive/allwind.csv')
df = df.drop(['ISTANBUL WindSpeed(m/s)'], axis = 1)
df = df.drop(['Unnamed: 0'] , axis = 1 )

df['Date'] = pd.date_range(start = '20180201' , freq = 'H' , periods = len(df))
df.dropna(inplace = True)
df.head()

# preprocessing:
rowsData = []
for i,row in tqdm(df.iterrows() , total = len(df)):

  data = dict(
      target = row.target_load,
      weekDay = row.Date.dayofweek,
      weekOfYear = row.Date.week,
      hourOfDay = row.Date.hour,
      dayOfMonth = row.Date.day,
      AMASYA_WindSpeed = row['AMASYA WindSpeed(m/s)'],
      KIRLARELIWindSpeed = row['KIRLARELI WindSpeed(m/s)'],
      TEKIRDAGWindSpeed = row['TEKIRDAG WindSpeed(m/s)'],
      EDIRNEWindSpeed = row['EDIRNE WindSpeed(m/s)'],
      BURSAWindSpeed = row['BURSA WindSpeed(m/s)'],
      SIVASWindSpeed = row['SIVAS WindSpeed(m/s)'],
      BILECIKWindSpeed = row['BILECIK WindSpeed(m/s)'],
      AYDINWindSpeed = row['AYDIN WindSpeed(m/s)'],
      HATAYWindSpeed = row['HATAY WindSpeed(m/s)'],
      TOKATWindSpeed = row['TOKAT WindSpeed(m/s)'],
      KONYAWindSpeed = row['KONYA WindSpeed(m/s)'],
      GAZIANTEPWindSpeed = row['GAZIANTEP WindSpeed(m/s)'],
      OSMANIYEWindSpeed = row['OSMANIYE WindSpeed(m/s)'],
      CNKWindSpeed = row['CNK WindSpeed(m/s)'],
      MUGLAWindSpeed = row['MUGLA WindSpeed(m/s)'],
      YALOVAWindSpeed = row['YALOVA WindSpeed(m/s)'],
      ADIYAMANWindSpeed = row['ADIYAMAN WindSpeed(m/s)'],
      KOCWindSpeed = row['KOC WindSpeed(m/s)'],
      KAYSERIWindSpeed = row['KAYSERI WindSpeed(m/s)'],
      ISPARTAWindSpeed = row['ISPARTA WindSpeed(m/s)'],
      USAKWindSpeed = row['USAK WindSpeed(m/s)'],
      MERSINWindSpeed = row['MERSIN WindSpeed(m/s)'],
      IZMIRWindSpeed = row['IZMIR WindSpeed(m/s)'],
      MANISAWindSpeed = row['MANISA WindSpeed(m/s)'],
      AFYONWindSpeed = row['AFYON WindSpeed(m/s)']
)
  
  rowsData.append(data)
df_data = pd.DataFrame(rowsData)

df_data.head()

train_size = int(len(df_data) *0.85)
train_df, test_df = df_data[:train_size] , df_data[train_size + 1:]
train_df.shape , test_df.shape

scaler = MinMaxScaler(feature_range=(-1,1))
scaler = scaler.fit(train_df)

train_df = pd.DataFrame(
    scaler.transform(train_df),
    index = train_df.index,
    columns = train_df.columns
)
test_df = pd.DataFrame(
    scaler.transform(test_df),
    index = test_df.index,
    columns = test_df.columns
)

# sequence initialization:

def create_sequence(input_data: pd.DataFrame, target_variable , seq_length):
  sequences = []
  data_size = len(input_data)
  #[1,2,3,5,4,7,6,8]
  for i in tqdm(range(data_size - seq_length)):
    
    sequence = input_data[i:i+seq_length]
    label_position = i+seq_length
    label = input_data.iloc[label_position][target_variable]
    sequences.append((sequence,label))
  return sequences

train_seq = create_sequence(train_df , 'target' , seq_length = 144)

test_seq = create_sequence(test_df , 'target' , seq_length=144)

train_seq[0][0]

