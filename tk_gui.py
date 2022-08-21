from tkinter import *
from tkinter import ttk

def execute():
    value = combobox.get()
    if value == 'Video Test':
        message = 'Testing video feed...'
    elif value == 'Calibration':
        message = 'Calibrating...'
    elif value == 'Main Pipeline':
        message = 'Executing main routine...'
    if message:
        text['state'] = 'normal'
        text.insert(END, '\n'+message)
        text['state'] = 'disabled'
        text.see('end')

root = Tk()
#root.attributes('-fullscreen', True)
root.geometry("1920x1080")

frame = ttk.Frame(root, width=1100, height=825)
#frame = ttk.Frame(root)
frame['borderwidth'] = 2
frame['relief'] = 'sunken'
frame['padding'] = 5 
#frame.pack(side = 'left', fill='y', padx=10, pady=10)
frame.grid(column=0, row=0, padx=10, pady=10)
frame.pack_propagate(0)

frame2 = ttk.Frame(root, width=250, height=825)
frame2['borderwidth'] = 2
frame2['relief'] = 'sunken'
frame2['padding'] = 5 
frame2.grid(column=1, row=0, pady=10)
# #frame.grid(column=0,row=0, padx = 10, pady=10)
frame2.pack_propagate(0)

my_label = Label(frame, text='You are going to be beautiful')
my_label.pack()

current_var = StringVar()
combobox = ttk.Combobox(frame2, textvariable=current_var)
combobox['values'] = ('Video Test', 'Calibration', 'Main Pipeline')
combobox['state'] = 'readonly'
combobox.current(0)
combobox.pack(side='top')
combobox.bind("<<ComboboxSelected>>",lambda e: frame2.focus())

exit_button = Button(frame2, text="Exit", command=root.destroy)
exit_button.pack(side='bottom')

text = Text(frame2, height=15, width=35, bg = 'black', fg = 'green2', padx=5, pady=5, yscrollcommand=True)
text.pack(side = 'bottom')
text.insert('1.0', 'Welcome!')
text['state'] = 'disabled'

execute_button = Button(frame2, text="Execute", command = execute)
execute_button.pack(side='top')

frame3 = ttk.Frame(frame2)
frame3['borderwidth'] = 2
#frame3['relief'] = 'sunken'
frame3['padding'] = 5 
frame3.pack(side = 'bottom')
#frame3.pack_propagate(0)

right_arrow = PhotoImage(file = "assets/right-arrow.png")
right_arrow = right_arrow.subsample(15,15)
rarrow = Button(frame3, text = 'Y+', image = right_arrow).pack(side = 'right')

left_arrow = PhotoImage(file = "assets/left-arrow.png")
left_arrow = left_arrow.subsample(15,15)
larrow = Button(frame3, text = 'Y-', image = left_arrow).pack(side = 'left')

top_arrow = PhotoImage(file = "assets/up-arrow.png")
top_arrow = top_arrow.subsample(15,15)
tarrow = Button(frame3, text = 'X-', image = top_arrow).pack(side = 'top')

down_arrow = PhotoImage(file = "assets/down-arrow.png")
down_arrow = down_arrow.subsample(15,15)
darrow = Button(frame3, text = 'X+', image = down_arrow).pack(side = 'bottom')

z_down_arrow = PhotoImage(file = "assets/z-down.png")
z_down_arrow = z_down_arrow.subsample(15,15)
zdarrow = Button(frame3, text = 'Z-', image = z_down_arrow).pack(side = 'bottom')

z_up_arrow = PhotoImage(file = "assets/z-up.png")
z_up_arrow = z_up_arrow.subsample(15,15)
zuarrow = Button(frame3, text = 'Z-', image = z_up_arrow).pack(side = 'top')

root.mainloop()
