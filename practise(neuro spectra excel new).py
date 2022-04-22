import mne
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.signal as sp
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_file(file_name, channels=False):
    os.chdir(r"C:\Users\kosti_000\Desktop\Python\Projects\practise bio\files")
    data=mne.io.read_raw_edf(file_name,preload=True)
    dataraw=data.get_data()
    channels=data.ch_names
    events=np.genfromtxt(file_name[:-4]+".ann", dtype=int,delimiter=",",skip_header=1)
    if channels == True:    
        return channels
    else:
        return dataraw,events

data,events=load_file("data.edf")
channels = load_file("data.edf",channels = True)

def FFT(data, window = 500, freqs = False): # def возвращает спектр сигнала, dat - сигнал, freqs  - надо ли возвращать спектр частот
    spectre = abs(np.fft.fft(data).real)**2
    freq = np.fft.fftfreq(window, d  = 0.004) # параметр 500 - время участка для анализа в отсчётах
    if freqs == True:
        return freq , spectre
    else:
        return spectre

def Epoched_data(data, window):
    e_data=data[:,events[0,0]:events[0,0]+window]
    for i in events[1:,0]:
        e_data=np.append(e_data, data[:,i:i+window], axis=1)
    return e_data[:-4,:].reshape(-1, 19,500)
#e_data(метки, каналы, данные)

def Feat_spectre(e_data): 
    spectre = np.zeros([1,e_data.shape[1]])
    freqs = np.zeros([250])
    for i in range(e_data.shape[0]):
        spec=np.array(FFT(e_data[i,0,:])[:250].reshape(250,1))
        for j in range(1,e_data.shape[1]):   
            spec=np.append(spec, np.array(FFT(e_data[i,j,:])[:250].reshape(250,1)) ,axis=1)
        spectre = np.append(spectre, spec,axis=0)
    spectre=spectre[1:,:]
    for i in range(e_data.shape[0]):    
        freqs= np.append(freqs,np.array(FFT(e_data[i,0,:],True)[0][:250]))
    freqs=freqs[250:]
    return spectre, freqs

e_data = Epoched_data(data, 500)
'''
spectre = Feat_spectre(e_data)[0]
freqs = Feat_spectre(e_data)[1]

series = events[:,2].repeat(250).reshape(250*e_data.shape[0],1)
marks = freqs.reshape(250*e_data.shape[0],1)
series = np.append(series,marks,axis=1)
series = np.append(series , spectre, axis=1)

np.savetxt('test.csv', series , delimiter=',', fmt='%s')
'''

spectre = Feat_spectre(e_data)[0]
ev = list(events[:,2])*250
ev = np.array(ev).reshape(250,-1)
ev = ev.T.reshape(-1,1)
freqs = FFT(Epoched_data(data,500),freqs = True)[0][:250]
freqs1 = np.array(list(freqs)*118).reshape(-1,1)
db  = pd.DataFrame(spectre) # создаем датафрейм, записываем туда спектры
db.insert(19, "marks",ev , True) # записываем в конец номера меток
db.insert(0, "freqs",freqs1 , True) # записываем в начало частоты
channels1 = ['freqs','EEG F7-A1', 'EEG F3-A1', 'EEG Fz-A1', 'EEG F4-A2', 'EEG F8-A2', 'EEG T3-A1', 'EEG C3-A1', 'EEG Cz-A2', 'EEG C4-A2', 'EEG T4-A2', 'EEG T5-A1', 'EEG P3-A1', 'EEG Pz-A1', 'EEG P4-A2', 'EEG T6-A2', 'EEG O1-A1', 'EEG O2-A2', 'EEG A1-A2', 'EEG A1-N','marks']
db.columns = channels1 #добавляем названия каналов(даны выше)
# print(db)
db_1cut = db.iloc[:50,:5]
db_1cut.insert(5,'marks', ev[:50])

db_2cut=db.iloc[4250:4300,:5]
db_2cut.insert(5,'marks', ev[4250:4300])

df=pd.concat([db_1cut,db_2cut], ignore_index=True, axis=0)
X=df.iloc[:,1:5]
y=df.iloc[:,5]
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3)
ref=RandomForestClassifier()
ref.fit(X_train, y_train)#левая правая
print(ref.score(X_test, y_test))

# print(db_cut)
# sns.pairplot(db_cut,hue = "marks") # строим с ними диаграмму рассеяния(комп не потянет)
# db = pd.read_csv('test2.csv', names  = ('marks','freqs')+channels)
# print(db)

#sns.pairplot(pd.DataFrame(db.iloc[:5000,:]) ,hue = ",marks")

#np.savetxt('test3.csv', spectre , delimiter=',', fmt='%s')















































