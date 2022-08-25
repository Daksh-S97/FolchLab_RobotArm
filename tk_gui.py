from tkinter import *
from tkinter import ttk
from customtkinter import *

set_appearance_mode("System")
set_default_color_theme("green")

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

#root = Tk()
root = CTk()

#root.attributes('-fullscreen', True)
root.geometry("1920x1080")

#frame = ttk.Frame(root, width=1100, height=825)
frame = CTkFrame(root, width=1100, height=825)
#frame = ttk.Frame(root)
frame['borderwidth'] = 2
frame['relief'] = 'sunken'
frame['padding'] = 5 
#frame.pack(side = 'left', fill='y', padx=10, pady=10)
frame.grid(column=0, row=0, padx=10, pady=10)
frame.pack_propagate(0)

#frame2 = ttk.Frame(root, width=250, height=825)
frame2 = CTkFrame(root, width=250, height=825)
frame2['borderwidth'] = 2
frame2['relief'] = 'sunken'
frame2['padding'] = 5 
frame2.grid(column=1, row=0, pady=10)
# #frame.grid(column=0,row=0, padx = 10, pady=10)
frame2.pack_propagate(0)

my_label = CTkLabel(frame, text='You are going to be beautiful')
my_label.pack()

current_var = StringVar()
#combobox = CTkComboBox(frame2, textvariable=current_var)
combobox = CTkComboBox(frame2, values = ['Video Test', 'Calibration', 'Main Pipeline'], state='readonly')
#combobox['values'] = ('Video Test', 'Calibration', 'Main Pipeline')
#combobox['state'] = 'readonly'
combobox.set('Video Test')
#combobox.current(0)
combobox.pack(side='top', pady = 10)
combobox.bind("<<ComboboxSelected>>",lambda e: frame2.focus())

exit_button = CTkButton(frame2, text="Exit", command=root.destroy)
exit_button.pack(side='bottom', pady = 10)

text = Text(frame2, height=15, width=35, bg = 'black', fg = 'green2', padx=5, pady=5, yscrollcommand=True)
text.pack(side = 'bottom')
text.insert('1.0', 'Welcome!')
text['state'] = 'disabled'

execute_button =CTkButton(frame2, text="Execute", command = execute)
execute_button.pack(side='top')

frame3 = CTkFrame(frame2)
frame3['borderwidth'] = 2
#frame3['relief'] = 'sunken'
frame3['padding'] = 5 
frame3.pack(side = 'bottom', pady = 10)
#frame3.pack_propagate(0)

right_arrow = PhotoImage(file = "assets/right-arrow.png")
right_arrow = right_arrow.subsample(15,15)
rarrow = CTkButton(frame3, text = '', width=45, height=45, image = right_arrow).pack(side = 'right', padx = 3, pady = 3)

left_arrow = PhotoImage(file = "assets/left-arrow.png")
left_arrow = left_arrow.subsample(15,15)
larrow = CTkButton(frame3, text = '', width=45, height=45,image = left_arrow).pack(side = 'left',  padx = 3, pady = 3)

top_arrow = PhotoImage(file = "assets/up-arrow.png")
top_arrow = top_arrow.subsample(15,15)
tarrow = CTkButton(frame3, text = '', width=45, height=45,image = top_arrow).pack(side = 'top',  padx = 3, pady = 3)

down_arrow = PhotoImage(file = "assets/down-arrow.png")
down_arrow = down_arrow.subsample(15,15)
darrow = CTkButton(frame3, text = '', width=45, height=45,image = down_arrow).pack(side = 'bottom',  padx = 3, pady = 3)

z_down_arrow = PhotoImage(file = "assets/z-down.png")
z_down_arrow = z_down_arrow.subsample(15,15)
zdarrow = CTkButton(frame3, text = '', width=45, height=45, image = z_down_arrow).pack(side = 'bottom',  padx = 3, pady = 3)

z_up_arrow = PhotoImage(file = "assets/z-up.png")
z_up_arrow = z_up_arrow.subsample(15,15)
zuarrow = CTkButton(frame3, text = '', width=45, height=45,image = z_up_arrow).pack(side = 'top',  padx = 3, pady = 3)

switch_var = StringVar(value="off")

def switch_event():
    state = switch_var.get()
    text['state'] = 'normal'
    text.insert(END, '\n'+f'Manual lock: {state.upper()}')
    text['state'] = 'disabled'
    text.see('end')

switch_1 = CTkSwitch(master=frame2, text="Manual Lock", command=switch_event,
                                   variable=switch_var, onvalue="on", offvalue="off")
switch_1.pack(padx=20, pady=10)

def slider_event(value):
    print(value)

slider = CTkSlider(master=frame2, from_=0, to=100, command=slider_event, orient='vertical', number_of_steps=100)
slider.pack(side = 'left')

root.mainloop()
