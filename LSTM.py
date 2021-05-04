import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size

df = pd.read_csv('demandData.csv')
df = df.drop('Unnamed: 0' , axis =1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = df.drop('tarih' , axis =1)
df = df.drop('saat' , axis =1)
data = df.demand.values.astype('float64')
trainSplit = int(len(data)*0.80)
train_data = data[:-trainSplit]
test_data = data[-trainSplit:]
print(len(train_data))
print(len(test_data))
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
train_data_normalized = train_data_normalized.to(device)
timeStamps = 44

##
def create_inout_sequences(input_data, timeStamps):
    inout_seq = []
    L = len(input_data)
    for i in range(L-timeStamps):
        train_seq = input_data[i:i+timeStamps]
        train_label = input_data[i+timeStamps:i+timeStamps+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq
train_inout_seq = create_inout_sequences(train_data_normalized, timeStamps)
train_inout_seq[:3]
class LSTM(nn.Module):
    def __init__(self, input_size =1, hidden_layer = 1026):
        super().__init__()
        self.hidden_layer = hidden_layer
        #nn.lstm needs two arguments: size and number of neurons that are fed
    #recursively
        self.lstm = nn.LSTM(i , hidden_layer)
    #for regression output of linear layer is 1.
        self.linear = nn.Linear(hidden_layer , 1)
    ##LSTM layers have 3 outputs Outputs: output, (h_n, c_n)
    #We have to initialize a hidden cell state so that we can can use it 
    #as an input for the next time_stamp 
    #LSTM algorithm accepts three inputs: previous hidden state, 
    #previous cell state and current input.
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer),
                            torch.zeros(1,1,self.hidden_layer))
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
    
model = LSTM()
model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)

epochs = 6

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    
    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
