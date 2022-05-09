# -*- coding: utf-8 -*-
"""
Created on Mon May  9 09:33:23 2022

@author: maryp
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:01:18 2022

@author: maryp
"""

import sounddevice as sd
import soundfile as sf

import numpy as np
import pandas as pd
from scipy.io.wavfile import write
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import FeatureExtract
import parselmouth

from tkinter import * 
from  tkinter import ttk
import tkinter.font as font

import pickle

from pysinewave import SineWave

from os import walk
from os.path import join
import random

mypath = "D://OneDrive//Documents//CDT//Year2//Training//Outreach//BeCurious//a_data"
file_paths = []

for path, subdirs, files in walk(mypath):
    for name in files:
        if name.endswith(".wav"):
            p = join(path, name)
            p = p.replace("\\", "//")
            file_paths.append(p)

filename = "D:\OneDrive\Documents\CDT\Year2\Training\Outreach\BeCurious\output.wav"

big_df = pd.read_csv("D:\\OneDrive\\Documents\\CDT\\Year2\\Training\\Outreach\\BeCurious\\a_data\\feats_w_preds.csv", index_col=0)
big_df['Age'] = big_df['Age'].replace("under14", "Under 14")
big_df['Age'] = big_df['Age'].replace("over50", "Over 50")

sex_df = pd.read_csv("D:\\OneDrive\\Documents\\CDT\\Year2\\Training\\Outreach\\BeCurious\\a_data\\sex_pred.csv", index_col=0)
sex_df['sex'] = sex_df['sex'].replace("w", "Girl")
sex_df['sex'] = sex_df['sex'].replace("m", "Boy")
sex_df['sex_pred'] = sex_df['sex_pred'].replace("w", "Girl")
sex_df['sex_pred'] = sex_df['sex_pred'].replace("m", "Boy")

model_path = "D:\\OneDrive\\Documents\\CDT\\Year2\\Training\\Outreach\\BeCurious\\model.sav"

model = pickle.load(open(model_path, 'rb'))

new_df = pd.DataFrame(columns=big_df.columns)

accuracies = []

y_pred = big_df['Pred']

y = big_df['Age']

new_acc = accuracy_score(y, y_pred)

accuracies.append(new_acc)

cm = confusion_matrix(y, y_pred, labels=model.classes_)

class_accs = cm.diagonal()/cm.sum(axis=1)

acc_df = pd.DataFrame(columns=["15-20", "21-30", "31-40", "41-50", "Over 50", "Under 14"])

acc_df.loc[len(acc_df)] = class_accs.tolist()

ages_count = pd.DataFrame(columns = ['Age', 'new'])
ages_count['Age'] = ["Under 14", "15-20", "21-30", "31-40", "41-50", "Over 50"]
ages_count['new'] = [0, 0, 0, 0, 0, 0]

ages_count = ages_count.set_index('Age')

n_ages = big_df['Age'].value_counts()

ages_count = pd.concat([ ages_count, n_ages], axis=1)

ages_count= ages_count.rename(columns={"Age": "starting"})

human_correct = 0
computer_correct = 0

human_correct_sex = 0
computer_correct_sex = 0

age_total = 0
sex_total = 0

px = 1/plt.rcParams['figure.dpi']

# initialise a window. 
root = Tk() 
root.config(background='white') 
#root.geometry("1000x700") 
root.attributes('-fullscreen', True)
root.title("Be Curious")
tabControl = ttk.Notebook(root)

width = root.winfo_screenwidth()
height = root.winfo_screenheight()

f = font.Font(family='Comic Sans MS', size = '24')
f_small = font.Font(family='Comic Sans MS', size = '16')

style = ttk.Style()

style.configure('big.TButton', font=('Comic Sans MS', 16))

style.configure("Treeview.Heading", font=('Comic Sans MS', 16))

style.configure('My.TFrame', background='white')

tab1 = ttk.Frame(tabControl, style='My.TFrame')
tab2 = ttk.Frame(tabControl, style='My.TFrame')
tab3 = ttk.Frame(tabControl, style='My.TFrame')
tab4 = ttk.Frame(tabControl, style='My.TFrame')
tab5 = ttk.Frame(tabControl, style='My.TFrame')

tabControl.add(tab1, text ='Record Audio')
tabControl.add(tab2, text ='Pitch')
tabControl.add(tab3, text ='Accuracy')
tabControl.add(tab5, text ='Beat the Computer')
tabControl.add(tab4, text ='Beat the Computer - Hard')
tabControl.pack(expand = 1, fill ="both")

l1 = Label(tab1, text="", font=f, bg='white')
l1.grid(row=2, column=0)

l2 = Label(tab1, text="", font=f, bg='white')
l2.grid(row=2, column=1)

fig1 = Figure(figsize=(width*0.5*px, height*0.6*px)) 
 
ax1 = fig1.add_subplot(111) 

graph1 = FigureCanvasTkAgg(fig1, master=tab1) 
graph1.get_tk_widget().grid(row=3, column=0)


fig2 = Figure(figsize=(width*0.5*px, height*0.6*px)) 
ax2 = fig2.add_subplot(111) 

graph2 = FigureCanvasTkAgg(fig2, master=tab1) 
graph2.get_tk_widget().grid(row=3, column=1)


fig3 = Figure(figsize=(width*0.8*px, height*0.7*px)) 
ax3 = fig3.add_subplot(111) 

graph3 = FigureCanvasTkAgg(fig3, master=tab2) 
graph3.get_tk_widget().grid(row=3, column=0)

frame1 = Frame(tab1)
frame1.grid(row=4, column=0)
feats_tbl1 = ttk.Treeview(frame1, style='big.TButton', height = 1)
feats_tbl1['columns'] = ('pitch', 'jitter', 'shimmer')

feats_tbl1.column('#0', width=0, stretch=NO)
feats_tbl1.column("pitch",anchor=CENTER, width=80)
feats_tbl1.column("jitter",anchor=CENTER,width=80)
feats_tbl1.column("shimmer",anchor=CENTER,width=80)


feats_tbl1.heading('#0', text='', anchor=CENTER)
feats_tbl1.heading("pitch",text="Pitch",anchor=CENTER)
feats_tbl1.heading("shimmer",text="Shimmer",anchor=CENTER) 
feats_tbl1.heading("jitter",text="Jitter",anchor=CENTER)


feats_tbl1.grid(row=4, column=0)

frame2 = Frame(tab1)
frame2.grid(row=4, column=1)
feats_tbl2 = ttk.Treeview(frame2, style='big.TButton', height = 1)
feats_tbl2['columns'] = ('pitch', 'jitter', 'shimmer')

feats_tbl2.column("#0", width=0,  stretch=NO)
feats_tbl2.column("pitch",anchor=CENTER, width=80)
feats_tbl2.column("jitter",anchor=CENTER,width=80)
feats_tbl2.column("shimmer",anchor=CENTER,width=80)

feats_tbl1.heading('#0', text='', anchor=CENTER)
feats_tbl2.heading("#0",text="",anchor=CENTER)
feats_tbl2.heading("pitch",text="Pitch",anchor=CENTER)
feats_tbl2.heading("shimmer",text="Shimmer",anchor=CENTER)
feats_tbl2.heading("jitter",text="Jitter",anchor=CENTER)

feats_tbl2.grid(row=4, column=1)

def record_audio(fs = 22050, seconds = 3):
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write("D:\OneDrive\Documents\CDT\Year2\Training\Outreach\BeCurious\output.wav", fs, myrecording)  # Save as WAV file 

    
    return myrecording

def get_feats():
    recording = parselmouth.Sound("D:\OneDrive\Documents\CDT\Year2\Training\Outreach\BeCurious\output.wav")
    report = FeatureExtract.get_report(recording)
    
    features = FeatureExtract.get_feats(report, "test")
    return features      
        

def record_display_audio1():
    global feats1
    
    feats_tbl1.delete(*feats_tbl1.get_children())
    
    l1['text'] = ""
    l1['text'] = "3"
    l1.update_idletasks()
    time.sleep(1)
    l1['text'] = "2"
    l1.update_idletasks()
    time.sleep(1)
    l1['text'] = "1"
    l1.update_idletasks()
    time.sleep(1)
    l1['text'] = "Recording!"
    l1.update_idletasks()
    signal = record_audio()
    l1['text'] = "Done!"
    
    ax1.cla() 
    # ax.grid() 
    # ax.set_ylim(-1, 1)
    # ax.set_xlim(0, len(y))
    
    ax1.plot(signal)
    graph1.draw() 
    
    feats1 = get_feats()    
    
    feats_tbl1.insert(parent='',index='end',text='',
                   values=(feats1['Mean pitch'][0],feats1['Jitter (local)'][0],feats1['Shimmer (local)'][0]))
    
    feats_tbl1.grid(row=4, column=0)
    
def record_display_audio2():
    global feats2
    
    feats_tbl2.delete(*feats_tbl2.get_children())
    
    l2['text'] = "3"
    l2.update_idletasks()
    time.sleep(1)
    l2['text'] = "2"
    l2.update_idletasks()
    time.sleep(1)
    l2['text'] = "1"
    l2.update_idletasks()
    time.sleep(1)
    l2['text'] = "Recording!"
    l2.update_idletasks()
    signal = record_audio()
    l2['text'] = "Done!"
    
    ax2.cla() 
    # ax.grid() 
    # ax.set_ylim(-1, 1)
    # ax.set_xlim(0, len(y))
    
    ax2.plot(signal)
    graph2.draw() 
    
    feats2 = get_feats()    
    
    feats_tbl2.insert(parent='',index='end',text='',
                   values=(feats2['Mean pitch'][0],feats2['Jitter (local)'][0],feats2['Shimmer (local)'][0]))
    
    feats_tbl2.grid(row=4, column=0)
    
    
def make_prediction(feats):
    feats = feats.fillna(0)
    X = feats.drop("name", axis=1)
    
    pred_age = model.predict(X)
    
    pred_popupmsg(pred_age, feats)
    
def pred_popupmsg(prediction, data):
    global popup
    
    popup = Tk()
    popup.wm_title("Computer Guess")
    
    t_label = ttk.Label(popup, text="I think you are:", font=f)
    t_label.grid(row=0, column=0)
    
    pred_label = ttk.Label(popup, text=prediction[0], font=f)
    pred_label.grid(row=1, column=0)
    
    q_label = ttk.Label(popup, text="Was I right?", font=f)
    q_label.grid(row=2, column=0)
    
    yes_button = Button(popup, text="Yes", command = lambda: add_data(prediction[0], data, False), bg='green', font=f_small)
    yes_button.grid(row=3, column=0) 
    
    no_button = Button(popup, text="No", command = lambda: choose_age(data), bg='red', font=f_small)
    no_button.grid(row=4, column=0) 
    
    popup.mainloop()
    
def choose_age(data):
    global choice_popup
    
    choice_popup = Tk()
    choice_popup.wm_title("How old are you?")
    
    
    label = Label(choice_popup, text="Select age range", font=f)
    label.grid(row=0, column=0)
    B1 = Button(choice_popup, text="Under 14", command = lambda: add_data("Under 14", data), font=f)
    B1.grid(row=1, column=0)
    B2 = Button(choice_popup, text="15-20", command = lambda: add_data("15-20", data), font=f)
    B2.grid(row=1, column=1)
    B3 = Button(choice_popup, text="21-30", command = lambda: add_data("21-30", data), font=f)
    B3.grid(row=1, column=2)
    B4 = Button(choice_popup, text="31-40", command = lambda: add_data("31-40", data), font=f)
    B4.grid(row=1, column=3)
    B5 = Button(choice_popup, text="41-50", command = lambda: add_data("41-50", data), font=f)
    B5.grid(row=1, column=4)
    B6 = Button(choice_popup, text="Over 50", command = lambda: add_data("Over 50", data), font=f)
    B6.grid(row=1, column=5)
    
    choice_popup.mainloop()

    
def add_data(age, data, choice=True):
    global big_df
    global new_df
    global ages_count
    global choice_popup
    global popup
    
    data=data.drop('name', axis=1)
    data['Age'] = [age]
    data = data.fillna(0)
    big_df = big_df.append(data)
    new_df = new_df.append(data)
    ages_count.loc[age]['new'] += 1
    
    ax5.cla() 
    ages_count[['starting', 'new']].plot.bar(stacked=True, ax=ax5, rot=0)
    graph5.draw()
    
    if new_df.shape[0]%5 == 0:
        new_model()
    

    if choice == True:
        choice_popup.destroy()
    
    popup.destroy()

def new_model():
    global big_df
        
    y = big_df['Age']
    X = big_df.drop(['Age', 'File', 'Pred'], axis=1)
    
    model = DecisionTreeClassifier(max_depth=10).fit(X,y)
    
    y_pred = model.predict(X)
    
    new_acc = accuracy_score(y, y_pred)
    
    accuracies.append(new_acc)
    
    #Replot accuracy graphs
    
    ax4.plot(accuracies, color='k')
    graph4.draw() 
    
    matrix = confusion_matrix(y, y_pred)
    class_accs = matrix.diagonal()/matrix.sum(axis=1)
    
    acc_df.loc[len(acc_df)] = class_accs.tolist()
            
    ax6.cla() 
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
    disp.plot(ax=ax6, cmap='Blues', colorbar=False)
    
    ax7.cla()
    acc_df.plot(ax=ax7)
    graph7.draw()
    
    print("New model!")
    print(len(accuracies))
    

b1 = Button(tab1, text="Record", command=record_display_audio1, bg="red", fg="white", font=f) 
b1.grid(row=1, column=0)

b2 = Button(tab1, text="Record", command=record_display_audio2, bg="red", fg="white", font=f) 
b2.grid(row=1, column=1)

b3 = Button(tab1, text="Add", command= lambda: make_prediction(feats1), bg="blue", fg="white", font=f) 
b3.grid(row=5, column=0)

b4 = Button(tab1, text="Add", command= lambda: make_prediction(feats2), bg="blue", fg="white", font=f) 
b4.grid(row=5, column=1)


'''Tab 2 code'''

sinewave = SineWave(pitch = 10, pitch_per_second = 50, decibels_per_second = 10)

# slider current value
current_value = DoubleVar()
volume = DoubleVar()


def get_current_value(event):
    global sinewave
    
    n = current_value.get()/100
    v = volume.get()
    time = np.arange(0, 50, 0.1)
    amplitude = v*(np.sin(n*time))
    
    ax3.cla()     
    ax3.set_ylim(-1, 1)
    ax3.get_xaxis().set_visible(False)
    ax3.plot(time, amplitude)
    graph3.draw() 
    
    sinewave.set_pitch(n*10)
    sinewave.set_volume(v*10)
    
    
def play_audio():
    global sinewave

    sinewave.play()

def stop_audio():
    global sinewave
    
    sinewave.stop()

#  slider
f_frame = LabelFrame(tab2, text="Pitch", font=f)
f_frame.grid(column=0, row=0)

slider = ttk.Scale(f_frame, from_=10, to=200, orient='horizontal', command=get_current_value, variable=current_value, length=width*0.6)

slider.pack()
slider.set(100)

v_frame = LabelFrame(tab2, text="Volume", font=f)
v_frame.grid(column=1, row=3)

v_slider = ttk.Scale(v_frame, from_=1, to=0, orient='vertical', command=get_current_value, variable=volume, length=height*0.6)
v_slider.pack()
v_slider.set(0.5)

b_frame = LabelFrame(tab2, bg='white', borderwidth = 0, highlightthickness = 0)
b_frame.grid(column=0, row=4)

b5 = Button(b_frame, text="Play", command=play_audio, bg="blue", fg="white", font=f) 
b5.grid(row=0, column=0, padx=7)

b5 = Button(b_frame, text="Stop", command=stop_audio, bg="red", fg="white", font=f) 
b5.grid(row=0, column=1, padx=7)


'''Tab 3 code'''

graph_4_title = Label(tab3, text="Overall Accuracy", font=f, bg='white')
graph_4_title.grid(row=0, column=0)

fig4 = Figure(figsize=(width*0.4*px, height*0.4*px)) 
 
ax4 = fig4.add_subplot(111) 

graph4 = FigureCanvasTkAgg(fig4, master=tab3) 
graph4.get_tk_widget().grid(row=1, column=0)

graph_5_title = Label(tab3, text="Number of people in each group", font=f, bg='white')
graph_5_title.grid(row=0, column=1)

fig5 = Figure(figsize=(width*0.4*px, height*0.4*px)) 
 
ax5 = fig5.add_subplot(111) 

graph5 = FigureCanvasTkAgg(fig5, master=tab3) 
graph5.get_tk_widget().grid(row=1, column=1)

ages_count.plot.bar(stacked=True, ax=ax5, rot=0)

graph_6_title = Label(tab3, text="Accuracy for each group", font=f, bg='white')
graph_6_title.grid(row=2, column=0)

fig6 = Figure(figsize=(width*0.4*px, height*0.4*px)) 
 
ax6 = fig6.add_subplot(111) 

graph6 = FigureCanvasTkAgg(fig6, master=tab3) 
graph6.get_tk_widget().grid(row=3, column=1)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=ax6, cmap='Blues', colorbar=False)

graph_7_title = Label(tab3, text="How the computer guessed", font=f, bg='white')
graph_7_title.grid(row=2, column=1)

fig7 = Figure(figsize=(width*0.4*px, height*0.4*px)) 
 
ax7 = fig7.add_subplot(111) 

graph7 = FigureCanvasTkAgg(fig7, master=tab3) 
graph7.get_tk_widget().grid(row=3, column=0, pady=10)

acc_df.plot(ax=ax7)
graph7.draw()

"""Tab 4 code"""

def load_file():
    global filename
    
    filename = random.choice(file_paths)
    data, fs = sf.read(filename, dtype='float32')  
    sd.play(data, fs)
    
    pred_l['text']=""
    human_l['text']=""
    actual_l['text']=""
    
    pred_l2['text']=""
    human_l2['text']=""
    actual_l2['text']=""
    
def repeat_file():
    global filename
    data, fs = sf.read(filename, dtype='float32')  
    sd.play(data, fs)
    
def show_results(n):
    global filename
    global computer_correct
    global human_correct
    global age_total
    
    age_total+=1
    
    age_dict = {0: "Under 14", 1: "15-20", 2: "21-30", 3: "31-40", 4: "41-50", 5:"Over 50"}
    guess_val = age_dict[n]
    actual = big_df.loc[big_df['File'] == filename].Age.values[0]
    pred = big_df.loc[big_df['File'] == filename].Pred.values[0]
    
    pred_l["text"] = pred
    human_l ["text"] = guess_val
    actual_l["text"] = actual
    
    if pred == actual:
        computer_correct+=1
        pred_l.config(fg="green")
    else:
        pred_l.config(fg="red")
    
    if guess_val == actual:
        human_correct+=1
        human_l.config(fg="green")
    else:
        human_l.config(fg="red")
        
    score_l['text'] = str(computer_correct)+" - "+str(human_correct)
    total_l['text'] = "Total guessed - " + str(age_total)
    

guess_frame = LabelFrame(tab4, borderwidth=0, highlightthickness=0, bg='white')
guess_frame.grid(row=0, column=0, padx=8)

butt_frame = LabelFrame(guess_frame, borderwidth=0, highlightthickness=0, bg='white')
butt_frame.grid(row=0, column=0)    
gen_b = Button(butt_frame, text="Play File", command=load_file, bg="blue", fg="white", font=f) 
gen_b.grid(row=0, column=0, padx=8)
rep_b = Button(butt_frame, text="Repeat", command=repeat_file, bg="blue", fg="white", font=f) 
rep_b.grid(row=0, column=1, padx=8)

g_frame = LabelFrame(guess_frame, text="Guess age:", font=f, borderwidth=0, highlightthickness=0, bg='white')
g_frame.grid(row=1, column=0)

B1 = ttk.Button(g_frame, text="Under 14", command = lambda: show_results(0), style="big.TButton")
B1.grid(row=1, column=0)
B2 = ttk.Button(g_frame, text="15-20", command = lambda: show_results(1), style="big.TButton")
B2.grid(row=1, column=1)
B2 = ttk.Button(g_frame, text="21-30", command = lambda: show_results(2), style="big.TButton")
B2.grid(row=1, column=2)
B2 = ttk.Button(g_frame, text="31-40", command = lambda: show_results(3), style="big.TButton")
B2.grid(row=1, column=3)
B2 = ttk.Button(g_frame, text="41-50", command = lambda: show_results(4), style="big.TButton")
B2.grid(row=1, column=4)
B2 = ttk.Button(g_frame, text="Over 50", command = lambda: show_results(5), style="big.TButton")
B2.grid(row=1, column=5)


p_frame = LabelFrame(guess_frame, text="Computer thinks:", width=width*0.3, height=height*0.2, font=f, bg='white')
p_frame.grid(row=2, column=0)
p_frame.grid_propagate(False)

pred_l = Label(p_frame, text="", font=f, bg='white')
pred_l.grid(row=0, column=0)

h_frame = LabelFrame(guess_frame, text="Human thinks:", width=width*0.3, height=height*0.2, font=f, bg='white')
h_frame.grid(row=3, column=0)
h_frame.grid_propagate(False)

human_l = Label(h_frame, text="", font=f, bg='white')
human_l.grid(row=0, column=0)

a_frame = LabelFrame(guess_frame, text="Actual age:", width=width*0.3, height=height*0.2, font=f, bg='white')
a_frame.grid(row=4, column=0)
a_frame.grid_propagate(False)

actual_l = Label(a_frame, text="", font=f, bg='white')
actual_l.grid(row=0, column=0)

results_frame = LabelFrame(tab4, text="Leader board", width=width*0.25, height=height*0.3, font=f, bg='white')
results_frame.grid(row=0, column=1, padx=8)
results_frame.grid_propagate(False)

title_l = Label(results_frame, text="Computer - Human", font=f_small, bg='white')
title_l.grid(row=0, column=0)
score_l = Label(results_frame, text="", font=f, bg='white')
score_l.grid(row=1, column=0)

total_l = Label(results_frame, text="Total guessed - 0", font=f_small, bg='white')
total_l.grid(row=2, column=0, columnspan=1)


"""Tab 5 code"""

def show_results_sex(n):
    global filename
    global computer_correct_sex
    global human_correct_sex
    global sex_total
    
    sex_total+=1
    
    sex_dict = {0: "Girl", 1: "Boy"}
    guess_val = sex_dict[n]
    actual = sex_df.loc[sex_df['File'] == filename].sex.values[0]
    pred = sex_df.loc[sex_df['File'] == filename].sex_pred.values[0]
    
    pred_l2["text"] = pred
    human_l2["text"] = guess_val
    actual_l2["text"] = actual
    
    if pred == actual:
        computer_correct_sex+=1
        pred_l2.config(fg="green")
        
    else:
        pred_l2.config(fg="red")
    
    if guess_val == actual:
        human_correct_sex+=1
        human_l2.config(fg="green")
    else:
        human_l2.config(fg="red")
    
        
    score_l2['text'] = str(computer_correct_sex)+" - "+str(human_correct_sex)
    total_l_sex['text'] = "Total guessed - " + str(sex_total)

guess_frame_sex = LabelFrame(tab5, borderwidth=0, highlightthickness=0, bg="white")
guess_frame_sex.grid(row=0, column=0, padx=8)

butt_frame_sex = LabelFrame(guess_frame_sex, borderwidth=0, highlightthickness=0, bg="white")
butt_frame_sex.grid(row=0, column=0)    
gen_b_sex = Button(butt_frame_sex, text="Play File", command=load_file, bg="blue", fg="white", font=f) 
gen_b_sex.grid(row=0, column=0, padx=8)
rep_b_sex = Button(butt_frame_sex, text="Repeat", command=repeat_file, bg="blue", fg="white", font=f) 
rep_b_sex.grid(row=0, column=1, padx=8)

g_frame_sex = LabelFrame(guess_frame_sex, text="Guess gender:", font=f, borderwidth=0, highlightthickness=0, bg='white')
g_frame_sex.grid(row=1, column=0)

B1_sex = ttk.Button(g_frame_sex, text="Girl", command = lambda: show_results_sex(0), style="big.TButton")
B1_sex.grid(row=1, column=0)
B2_sex = ttk.Button(g_frame_sex, text="Boy", command = lambda: show_results_sex(1), style="big.TButton")
B2_sex.grid(row=1, column=1)


p_frame_sex = LabelFrame(guess_frame_sex, text="Computer thinks:", width=width*0.3, height=height*0.2, font=f, bg='white')
p_frame_sex.grid(row=2, column=0)
p_frame_sex.grid_propagate(False)

pred_l2 = Label(p_frame_sex, font=f, bg='white')
pred_l2.grid(row=0, column=0)

h_frame_sex = LabelFrame(guess_frame_sex, text="Human thinks:", width=width*0.3, height=height*0.2, font=f, bg='white')
h_frame_sex.grid(row=3, column=0)
h_frame_sex.grid_propagate(False)

human_l2 = Label(h_frame_sex, text="", font=f, bg='white')
human_l2.grid(row=0, column=0)

a_frame_sex = LabelFrame(guess_frame_sex, text="Actual gender:", width=width*0.3, height=height*0.2, font=f, bg='white')
a_frame_sex.grid(row=4, column=0)
a_frame_sex.grid_propagate(False)

actual_l2 = Label(a_frame_sex, text="", font=f, bg='white')
actual_l2.grid(row=0, column=0)

results_frame_sex = LabelFrame(tab5, text="Leader board", width=width*0.25, height=height*0.3, font=f, bg='white')
results_frame_sex.grid(row=0, column=1, padx=8)
results_frame_sex.grid_propagate(False)

title_l_sex = Label(results_frame_sex, text="Computer - Human", font=f_small, bg='white')
title_l_sex.grid(row=0, column=0)
score_l2 = Label(results_frame_sex, text="", font=f, bg='white')
score_l2.grid(row=1, column=0)

total_l_sex = Label(results_frame_sex, text="Total guessed - 0", font=f_small, bg='white')
total_l_sex.grid(row=2, column=0, columnspan=1)

root.mainloop()
