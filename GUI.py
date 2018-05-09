from tkinter import * 
from tkinter import filedialog
from tkinter import messagebox
from scipy import misc

import numpy as np
from neural_network import NeuralNetwork

Red = None
ready = False

def abrir(var):
    folder_selected = filedialog.askopenfilename()
    var.set(folder_selected)

def classify(image_path):
    global Red, ready
    if ready:
        image = misc.imread(image_path, flatten=True)
        image = image.flatten()
        print(f_act.get())
        t_1, t_2, t_3 = Red.test_image(image, f_act.get())
        messagebox.showinfo("Result", "The class of the image is "+str(t_1)+"\nAlso possible: "+str(t_2)+", "+str(t_3))
    else:
        messagebox.showerror("Error", "Load the weights first")
def load(lista, two_layers):
    global Red, ready
    two_l = False if two_layers == 0 else True

    w1 = np.load(lista[0])

    w3 = np.load(lista[2])

    if two_l:
        w2 = np.load(lista[1])
        if w1.shape[1] == w2.shape[0] and w2.shape[1] == w3.shape[0]:
            Red = NeuralNetwork(w1.shape[0], w2.shape[0], w3.shape[0], w3.shape[1], 0.0085, two_l, 0)
            Red.load_Ws(w1,w2,w3)
            messagebox.showinfo("", "Weights loaded")
            ready = True

        else:
            messagebox.showerror("Error", "Number of neurons do not match")

    else:
        if w1.shape[1] == w3.shape[0]:
            Red = NeuralNetwork(w1.shape[0],0, w3.shape[0], w3.shape[1], 0.0085, two_l, 0)
            Red.load_Ws(w1,np.zeros(1),w3)
            messagebox.showinfo("", "Weights loaded")
            ready = True
        else:
            messagebox.showerror("Error", "Number of neurons do not match")


window=Tk()
window.geometry("400x300+20+20")
window.title("NeuralNetwork")


##EntryBox
W1 = StringVar()
entryBoxW1 = Entry(window,width=45,textvariable=W1).place(x=10,y=10)

W2 = StringVar()
entryBoxW2 = Entry(window,width=45,textvariable=W2).place(x=10,y=60)

W3 = StringVar()
entryBoxW3 = Entry(window,width=45,textvariable=W3).place(x=10,y=110)

twoL = IntVar()
Checkbutton(window, text="Two hidden layers", variable=twoL).place(x=80,y=160)

f_act = IntVar()


test = StringVar()
entryBoxTest = Entry(window,width=45,textvariable=test).place(x=10,y=210)

##Buttons 
bW1=Button(window,text="Select W1", command=lambda: abrir(W1)).place(x=300,y=10)
bW2=Button(window,text="Select W2", command=lambda: abrir(W2)).place(x=300,y=60)
bW3=Button(window,text="Select W3", command=lambda: abrir(W3)).place(x=300,y=110)
bImageTest=Button(window,text="Select image", command=lambda: abrir(test)).place(x=300,y=210)
bLoad=Button(window,text="Load weights",command=lambda:load([W1.get(),W2.get(),W3.get()], twoL.get())).place(x=240, y =160)
bClass=Button(window,text="Classify image",command=lambda:classify(test.get())).place(x=160, y = 250)

R1 = Radiobutton(window, text="ReLU", variable=f_act, value=0, anchor = W).place(x=60, y = 240)

R2 = Radiobutton(window, text="Sigmoid", variable=f_act, value=1, anchor = W).place(x=60, y = 260)




window.mainloop()


