from tkinter import *
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
from deps import cv_core
import numpy as np

class RobotApp:
    def __init__(self, root, window_title) -> None:
        self.root = root
        self.root.title(window_title)
        self.cap_on = False
        self.cap = None

        self.root.geometry("1920x1080")

        frame = ttk.Frame(root, width=1332, height=999)
        frame['borderwidth'] = 2
        frame['relief'] = 'sunken'
        frame['padding'] = 5 
        frame.grid(column=0, row=0, padx=10, pady=10)
        frame.pack_propagate(0)

        self.canvas = Canvas(frame, width = 1332, height = 999)
        self.canvas.pack()

        frame2 = ttk.Frame(root, width=250, height=825)
        #frame2 = CTkFrame(root, width=250, height=825)
        frame2['borderwidth'] = 2
        frame2['relief'] = 'sunken'
        frame2['padding'] = 5 
        frame2.grid(column=1, row=0, pady=10)
        # #frame.grid(column=0,row=0, padx = 10, pady=10)
        frame2.pack_propagate(0)

        current_var = StringVar()
        #combobox = CTkComboBox(frame2, textvariable=current_var)
        self.combobox = ttk.Combobox(frame2, values = ['Video Test', 'Calibration', 'Main Pipeline'], state='readonly')
        #combobox['values'] = ('Video Test', 'Calibration', 'Main Pipeline')
        #combobox['state'] = 'readonly'
        self.combobox.set('Video Test')
        #combobox.current(0)
        self.combobox.pack(side='top', pady = 10)
        self.combobox.bind("<<ComboboxSelected>>",lambda e: frame2.focus())

        exit_button = Button(frame2, text="Exit", command=root.destroy)
        exit_button.pack(side='bottom', pady = 10)

        self.text = Text(frame2, height=15, width=35, bg = 'black', fg = 'green2', padx=5, pady=5, yscrollcommand=True)
        self.text.pack(side = 'bottom')
        self.text.insert('1.0', 'Welcome!')
        self.text['state'] = 'disabled'

        execute_button = Button(frame2, text="Execute", command = self.execute)
        execute_button.pack(side='top')

        frame3 = ttk.Frame(frame2)
        frame3['borderwidth'] = 2
        #frame3['relief'] = 'sunken'
        frame3['padding'] = 5 
        frame3.pack(side = 'bottom', pady = 10)
        #frame3.pack_propagate(0)

        right_arrow = PhotoImage(file = "assets/right-arrow.png")
        right_arrow = right_arrow.subsample(15,15)
        rarrow = Button(frame3, text = '', width=45, height=45, image = right_arrow).pack(side = 'right', padx = 3, pady = 3)

        left_arrow = PhotoImage(file = "assets/left-arrow.png")
        left_arrow = left_arrow.subsample(15,15)
        larrow = Button(frame3, text = '', width=45, height=45,image = left_arrow).pack(side = 'left',  padx = 3, pady = 3)

        top_arrow = PhotoImage(file = "assets/up-arrow.png")
        top_arrow = top_arrow.subsample(15,15)
        tarrow = Button(frame3, text = '', width=45, height=45,image = top_arrow).pack(side = 'top',  padx = 3, pady = 3)

        down_arrow = PhotoImage(file = "assets/down-arrow.png")
        down_arrow = down_arrow.subsample(15,15)
        darrow = Button(frame3, text = '', width=45, height=45,image = down_arrow).pack(side = 'bottom',  padx = 3, pady = 3)

        z_down_arrow = PhotoImage(file = "assets/z-down.png")
        z_down_arrow = z_down_arrow.subsample(15,15)
        zdarrow = Button(frame3, text = '', width=45, height=45, image = z_down_arrow).pack(side = 'bottom',  padx = 3, pady = 3)

        z_up_arrow = PhotoImage(file = "assets/z-up.png")
        z_up_arrow = z_up_arrow.subsample(15,15)
        zuarrow = Button(frame3, text = '', width=45, height=45,image = z_up_arrow).pack(side = 'top',  padx = 3, pady = 3)

        self.root.mainloop()

    def execute(self):
        if not self.cap_on:
            value = self.combobox.get()
            if value == 'Video Test':
                message = 'Testing video feed...'
                self.start_vid(self.video_test)
            elif value == 'Calibration':
                message = 'Calibrating...'
            elif value == 'Main Pipeline':
                message = 'Executing main routine...'
                self.start_vid(self.main_pipe)
            if message:
                self.text['state'] = 'normal'
                self.text.insert(END, '\n'+message)
                self.text['state'] = 'disabled'
                self.text.see('end')
        else:
            self.stop_vid()

    def start_vid(self, func):
        self.stop_vid()
        self.cap_on = True
        self.cap = cv2.VideoCapture(0) 
        self.cap = cv_core.set_res(self.cap, cv_core.camera_res_dict['1200'])
        func()

    def stop_vid(self):
        self.cap_on = False
        
        if self.cap:
            self.cap.release()
        self.canvas.delete('all')

    def video_test(self):
        # Get a frame from the video source
        if self.cap_on:
            ret, frame = self.cap.read()

            if ret:
                frame = cv2.resize(frame, (1332,999), interpolation = cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = NW)

            self.root.after(10, self.video_test)

    def main_pipe(self):
        cont = cv_core.Contours()
        cameraMatrix = np.load('./cam_matrices/cam_mtx.npy')
        dist = np.load('./cam_matrices/dist.npy')
        newCameraMatrix = np.load('./cam_matrices/newcam_mtx.npy')
        offset = 40
        if self.cap_on:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
                pt = cont.find_contours(frame, 10, offset)
                a,b,r = pt
                plot_img = frame.copy()
                cv2.circle(plot_img, (a, b), r, (3, 162, 255), 2)
                cv2.circle(plot_img, (a, b), 1, (0, 0, 255), 3)
                cv2.circle(plot_img, (a, b), r - offset, (0, 255, 0), 2)
                cv2.circle(plot_img, cont.big_circ[:2], int(cont.big_circ[2]), (0, 0, 255), 2)
                # cv2.putText(plot_img, f"{cont.big_circ[2]*2}px = 60mm", (cont.big_circ[0]-25, cont.big_circ[1] - cont.big_circ[2] - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                
                cv2.drawContours(plot_img, cont.singular, -1,(0,255, 0),2)
                cv2.drawContours(plot_img, cont.clusters, -1,(0,0,255),2)
                for c in cont.singular:
                    # compute the center of the contour
                    M = cv2.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # draw the contour and center of the shape on the image
                    #cv2.drawContours(plot_img, [c], -1, (0, 255, 0), 2)
                    cv2.circle(plot_img, (cX, cY), 2, (0, 0, 255), -1)
                    cv2.putText(plot_img, f"{cX},{cY}", (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for c in cont.clusters:
                    # compute the center of the contour
                    M = cv2.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # draw the contour and center of the shape on the image
                    #cv2.drawContours(plot_img, [c], -1, (0, 255, 0), 2)
                    cv2.circle(plot_img, (cX, cY), 2, (0, 0, 255), -1)

                plot_img = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
                plot_img = cv2.resize(plot_img, (1332,999), interpolation = cv2.INTER_AREA)
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(plot_img))
                self.canvas.create_image(0, 0, image = self.photo, anchor = NW)

            self.root.after(10, self.main_pipe)


RobotApp(Tk(), "Tkinter and OpenCV")