import mne
import matplotlib.pyplot as plt
import os
import numpy as np

print(os.getcwd())
os.chdir(r"C:\Users\kosti_000\Desktop\Python\Projects\practise bio\files")
name="data.edf"
data=mne.io.read_raw_edf(name,preload=True)
dataraw=data.get_data()
channel_names=data.ch_names
# plt.plot(dataraw[15,:10000])
# plt.show()

events=np.genfromtxt("data.ann",dtype=int,delimiter=",",skip_header=1)
# print(events)
# print(events.shape)

l=dataraw.shape[1] #число отсчетов
eventchannel=np.zeros(l)
eventchannel_1 = np.zeros(l)
for i in range(events.shape[0]):#кол-во меток
    eventchannel[events[i,0]]=events[i,2]#заполняем канал меток начала
    eventchannel_1[events[i,0]+500]=events[i,2]#заполняем канал меток конца
   
# plt.plot(eventchannel) 
#plt.plot(eventchannel_1) 
# plt.show()
evch_start  = eventchannel.reshape(1,-1) # метка начала движений 
evch_end = eventchannel_1.reshape(1,-1) # метка окончаний движений
# print(eventchannel)
# print(eventchannel.shape)
# print(evch_start)
# print(evch_start.shape)

metadata = np.append(dataraw, evch_start, axis = 0)
metadata = np.append(metadata, evch_end, axis = 0)
# print(metadata.shape)
def T_axis_marks(series):
    taxis_marks = []
    for i in range(metadata.shape[1]):
        if metadata[-2,i] == series:#номер движения
            taxis_marks.append(i)
            taxis_marks.append(i+500)
    taxis_marks = np.array(taxis_marks).reshape(-1,2)
    return taxis_marks
# print(T_axis_marks(2))

def FFT(dat, freqs = False): # def возвращает спектр сигнала, dat - сигнал, freqs  - надо ли возвращать спектр частот
    spectre = abs(np.fft.fft(dat).real)**2
    freq = np.fft.fftfreq(500, d  = 0.004) # параметр 500 - время участка для анализа в отсчётах
    if freqs == True:
        return freq , spectre
    else:
        return spectre

def MOVE(n):
    marks=T_axis_marks(n)
    plt.Figure()
    plt.figure(figsize=(20,16))
    for j in range(1,dataraw.shape[0]):
        spectre=FFT(dataraw[j,marks[0,0]:marks[0,1]], True)
        freqs=spectre[0]# массив частот(для построения графика (x))
        spectre=FFT(dataraw[j,marks[0,0]:marks[0,1]]).reshape(1,-1)#транспонируем вектор
        for i in range(1,marks.shape[0]):
            signal = FFT(dataraw[j,marks[i,0]:marks[i,1]]).reshape(1,-1)#формируем участок для анализа
            spectre=np.append(signal,spectre,axis=0) 
        mean_spectre=np.mean(spectre,axis=0)*1E8
        plt.xlim(0,125)
        plt.ylim(0,40)
        plt.subplot(6,4,j)
        plt.plot(freqs,mean_spectre,color='red',linewidth=1)
        plt.grid()
        plt.title(channel_names[j])
    plt.subplots_adjust(wspace=.5,hspace=.5)
    plt.savefig('filemove.jpg')
    plt.show()

def MENTAL_MOVE(n):#4-6 серия это мысленное движение на 2-4 с вперед (+500);7-9 серия за 2 с(-500) до метки
    k=1 if (n<=6)&(n>=4) else -1 if (n>=7)&(n<=9) else 0
    if (k==0):
        q=0
    else:
        q=1
    marks=T_axis_marks(n)
    plt.Figure()
    plt.figure(figsize=(20,16))
    for j in range(1,dataraw.shape[0]):
        spectre=FFT(dataraw[j,marks[0,0]+500*k:marks[0,1]+500*k], True)
        freqs=spectre[0]# массив частот(для построения графика (x))
        spectre=FFT(dataraw[j,marks[0,0]:marks[0,1]]).reshape(1,-1)#транспонируем вектор
        for i in range(1,marks.shape[0]):
            signal = FFT(dataraw[j,marks[i,0]+500*k:marks[i,1]+500*k]).reshape(1,-1)#формируем участок для анализа
            spectre=np.append(signal,spectre,axis=0) 
        mean_spectre=np.mean(spectre,axis=0)*1E8*q
        plt.xlim(0,125)
        plt.ylim(0,40)
        plt.subplot(6,4,j)
        plt.plot(freqs,mean_spectre,color='blue',linewidth=1)
        plt.grid()
        plt.title(channel_names[j])
    plt.subplots_adjust(wspace=.5,hspace=.5)
    plt.savefig('filemental.jpg')
    plt.show()
    
def PREPARE_MOVE(n):#1-6 серия подготовка к движению(процесс отдавания команды мозгом) 1 с до метки(-250)
    k=1 if (n<=6)&(n>=1) else 0
    marks=T_axis_marks(n)
    plt.Figure()
    plt.figure(figsize=(20,16))
    for j in range(1,dataraw.shape[0]):
        spectre=FFT(dataraw[j,marks[0,0]-500:marks[0,1]-500], True)
        freqs=spectre[0]# массив частот(для построения графика (x))
        spectre=FFT(dataraw[j,marks[0,0]:marks[0,1]]).reshape(1,-1)#транспонируем вектор
        for i in range(1,marks.shape[0]):
            signal = FFT(dataraw[j,marks[i,0]-500:marks[i,1]-500]).reshape(1,-1)#формируем участок для анализа
            spectre=np.append(signal,spectre,axis=0) 
        mean_spectre=np.mean(spectre,axis=0)*1E8*k
        plt.xlim(0,125)
        plt.ylim(0,40)
        plt.subplot(6,4,j)
        plt.plot(freqs,mean_spectre,color='black',linewidth=1)
        plt.grid()
        plt.title(channel_names[j])
    plt.subplots_adjust(wspace=.5,hspace=.5)
    plt.savefig('fileprep.jpg')
    plt.show()

def FON():
    marks=T_axis_marks(10)
    plt.Figure()
    plt.figure(figsize=(20,16))
    signal=FFT(dataraw[0,marks[0,0]:marks[0,1]], True)
    freqs=signal[0]# массив частот(для построения графика (x))
    signal=signal[1]
    for j in range(1,dataraw.shape[0]): #цикл по каналам
        signal=np.append(signal,FFT(dataraw[j,marks[0,0]:marks[0,1]]),axis=0)
    first_mark=signal.reshape(-1,500)
    for j in range(1,dataraw.shape[0]): #цикл по каналам
        for k in range(1,30):
            spectre=FFT(dataraw[j,marks[0,0]+500*k:marks[0,1]+500*k]).reshape(1,-1)#транспонируем вектор
            spectre=np.append(first_mark[j,:].reshape(-1,500),spectre,axis=0) 
        mean_spectre=np.mean(spectre,axis=0)*1E8
        plt.xlim(0,125)
        plt.ylim(0,40)
        plt.subplot(6,4,j)
        plt.plot(freqs,mean_spectre,color='green',linewidth=1)
        plt.grid()
        plt.title(channel_names[j])
        plt.subplots_adjust(wspace=.5,hspace=.5)
    plt.savefig('filefon.jpg')
    plt.show()
FON()

def MOVE_1(n,marks,j):
    spectre=FFT(dataraw[j,marks[0,0]:marks[0,1]], True)
    freqs=spectre[0]# массив частот(для построения графика (x))
    spectre=FFT(dataraw[j,marks[0,0]:marks[0,1]]).reshape(1,-1)#транспонируем вектор
    for i in range(1,marks.shape[0]):
        signal = FFT(dataraw[j,marks[i,0]:marks[i,1]]).reshape(1,-1)#формируем участок для анализа
        spectre=np.append(signal,spectre,axis=0) 
    mean_spectre=np.mean(spectre,axis=0)*1E8
    return freqs, mean_spectre
    
    
def MENTAL_MOVE_1(n,marks,j):#4-6 серия это мысленное движение на 2-4 с вперед (+500);7-9 серия за 2 с(-500) до метки
    k=1 if (n<=6)&(n>=4) else -1 if (n>=7)&(n<=9) else 0
    if (k==0):
        q=0
    else:
        q=1
    spectre=FFT(dataraw[j,marks[0,0]+500*k:marks[0,1]+500*k], True)
    freqs=spectre[0]# массив частот(для построения графика (x))
    spectre=FFT(dataraw[j,marks[0,0]:marks[0,1]]).reshape(1,-1)#транспонируем вектор
    for i in range(1,marks.shape[0]):
        signal = FFT(dataraw[j,marks[i,0]+500*k:marks[i,1]+500*k]).reshape(1,-1)#формируем участок для анализа
        spectre=np.append(signal,spectre,axis=0) 
    mean_spectre=np.mean(spectre,axis=0)*1E8*q
    return freqs, mean_spectre

    
def PREPARE_MOVE_1(n,marks,j):#1-6 серия подготовка к движению(процесс отдавания команды мозгом) 1 с до метки(-250)
    k=1 if (n<=6)&(n>=1) else 0
    spectre=FFT(dataraw[j,marks[0,0]-500:marks[0,1]-500], True)
    freqs=spectre[0]# массив частот(для построения графика (x))
    spectre=FFT(dataraw[j,marks[0,0]:marks[0,1]]).reshape(1,-1)#транспонируем вектор
    for i in range(1,marks.shape[0]):
        signal = FFT(dataraw[j,marks[i,0]-500:marks[i,1]-500]).reshape(1,-1)#формируем участок для анализа
        spectre=np.append(signal,spectre,axis=0) 
    mean_spectre=np.mean(spectre,axis=0)*1E8*k
    return freqs, mean_spectre

def FON_1(j):
    marks=T_axis_marks(10)
    signal=FFT(dataraw[0,marks[0,0]:marks[0,1]], True)
    freqs=signal[0]# массив частот(для построения графика (x))
    signal=signal[1]
    for h in range(1,dataraw.shape[0]): #цикл по каналам
        signal=np.append(signal,FFT(dataraw[h,marks[0,0]:marks[0,1]]),axis=0)
    first_mark=signal.reshape(-1,500)
    for k in range(1,30):
        spectre=FFT(dataraw[j,marks[0,0]+500*k:marks[0,1]+500*k]).reshape(1,-1)#транспонируем вектор
        spectre=np.append(first_mark[j,:].reshape(-1,500),spectre,axis=0) 
    mean_spectre=np.mean(spectre,axis=0)*1E8
    return freqs, mean_spectre


def FULL(n):
    marks=T_axis_marks(n)
    plt.Figure()
    plt.figure(figsize=(20,16))
    for j in range(1,dataraw.shape[0]):
        plt.xlim(0,125)
        plt.ylim(0,40)
        plt.subplot(6,4,j)
        plt.plot(MOVE_1(n, marks, j)[0],MOVE_1(n, marks, j)[1],'r',linewidth=1)
        plt.plot(MENTAL_MOVE_1(n, marks, j)[0],MENTAL_MOVE_1(n, marks, j)[1],'blue',linewidth=1)
        plt.plot(PREPARE_MOVE_1(n, marks, j)[0],PREPARE_MOVE_1(n, marks, j)[1],'black',linewidth=1)
        plt.plot(FON_1(j)[0],FON_1(j)[1],'g',linewidth=1)
        plt.grid()
        plt.title(channel_names[j])
    plt.subplots_adjust(wspace=.5,hspace=.5)
    plt.savefig('file10.jpg')
    plt.show()

def SEPARATE(n):
    MOVE(n)
    MENTAL_MOVE(n)
    PREPARE_MOVE(n)
    FON()

# FULL(1)

# Движение - красный
# Мысленное движение - синее
# Подготовка к движению - черное
# Фон - зеленый   
    
#1-3 серия PREPARE_MOVE()-500,                       MOVE(), FON()
#4-6 серия PREPARE_MOVE()-500, MENTAL_MOVE()+500,    MOVE(), FON()
#7-9 серия MENTAL_MOVE()-500,                        MOVE(), FON()

# SEPARATE(4) 