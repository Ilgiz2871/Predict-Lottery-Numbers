import pandas as pd
import numpy as np
from   sklearn.preprocessing import StandardScaler
from   tensorflow.keras.models import Sequential
from   tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
import numpy as np

a = np.loadtxt('C:\\Temp\\loto.txt',dtype ='int')

df=pd.DataFrame(a,columns=list('ABCDEF'))

scaler=StandardScaler().fit(df.values)
transformed_dataset=scaler.transform(df.values)
transformed_df=pd.DataFrame(data=transformed_dataset, index=df.index)

number_of_rows = df.values.shape[0] 
window_length=7
number_of_features=df.values.shape[1]
print(number_of_rows)
print(number_of_features)

train = np.empty([number_of_rows-window_length,window_length,number_of_features], dtype=float)
label = np.empty([number_of_rows-window_length,number_of_features], dtype=float)
window_length=7

for i in range(0, number_of_rows-window_length):
    train[i]=transformed_df.iloc[i:i+window_length,0:number_of_features]
    label[i]=transformed_df.iloc[i+window_length:i+window_length+1,0:number_of_features]


bath_size = 100

model=Sequential()
model.add(Bidirectional(LSTM(240,input_shape=(window_length,number_of_features),return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240,input_shape=(window_length,number_of_features),return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240,input_shape=(window_length,number_of_features),return_sequences=True)))
model.add(Bidirectional(LSTM(240,input_shape=(window_length,number_of_features),return_sequences=False)))
model.add(Dense(45))
model.add(Dense(number_of_features))
model.compile(loss='mse',optimizer='rmsprop', metrics=['accuracy'])

model.fit(train, label, batch_size=100, epochs=300)

to_predict = np.loadtxt('C:\\Temp\\loto1.txt',dtype ='int')

#to_predict=np.array([[3,15,38,45,48,53],[1,8,12,16,18,42],[9,21,23,30,35,38],[2,8,36,51,52,54],[6,8,9,36,39,57]])
scaled_to_predict=scaler.transform(to_predict)

scaled_predicted_output_1=model.predict(np.array([scaled_to_predict]))

np.savetxt('C:\\Temp\\combination.npy',scaler.inverse_transform(scaled_predicted_output_1).astype(int)[0],fmt='%d')

print(scaler.inverse_transform(scaled_predicted_output_1).astype(int)[0])
