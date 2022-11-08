import tkinter as tk

master = tk.Tk()

def write_slogan():
    info_message = "Your message"
    tk.Label(master, text=info_message).grid(row=2, column=1) 

btn = tk.Button(master, text='ORDER Number', command=write_slogan)
btn.grid(row=3, column=1, sticky=tk.W, pady=4)

master.mainloop()