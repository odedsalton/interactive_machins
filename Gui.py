import tkinter
from tkinter import *

gloBla=0

def gui(song):
    # create a tkinter window
    root = Tk()
    frame = Frame(root)
    frame.pack()

    # root.geometry('200x200')
    Label(frame, text='The next song to play is:' + song, fg="white", bg="dark grey",
                  font="Verdana 12 bold").pack()

    def var1():
        global gloBla
        gloBla = 1
        root.destroy()

    def var0():
        global gloBla
        gloBla = 0
        root.destroy()

    # Create a Button
    btn1 = Button(frame, text='like !', fg="white", bg='dark green', bd='6', command=var1, width=15)
    btn2 = Button(frame, text='dislike !', fg="white", bg='dark red', bd='6', command=var0, width=15)

    btn1.pack(side=LEFT)
    btn2.pack(side=RIGHT)

    root.mainloop()

    return gloBla
