from tkinter import *
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
from deps import cv_core
import numpy as np
from deps.dobot_api import DobotApiDashboard, DobotApiMove
from deps import utils
import time

'''
plan: can I have different menu options that create/ destroy buttons that do different things?
can be one frame that has always-necessary commands such as enable/ disable, 
and maybe a menu can switch between buttons for calibration/ running/ etc(?) (what is etc?)
'''

'''CV2 WINDOW NOT WORKING IN TKINTER'''


class RobotApp:
    def __init__(self, root, window_title,Bot) -> None:
        self.root = root
        self.root.title(window_title)
        self.cap_on = False
        self.cap = None

        self.recorded = []
        self.anchorflag = 0

        #self.root.geometry("1920x1080")


        self.frame = ttk.Frame(self.root, width=250, height=425)
        self.frame['borderwidth'] = 2
        self.frame['relief'] = 'sunken'
        self.frame['padding'] = 5 
        self.frame.grid(column=1, row=1, padx=10, pady=10)
        self.frame.pack_propagate(0)


        '''self.canvas = Canvas(frame, width = 1332, height = 999)
        self.canvas.pack()'''


        frame2 = ttk.Frame(self.root, width=250, height=425)
        #frame2 = CTkFrame(root, width=250, height=825)
        frame2['borderwidth'] = 2
        frame2['relief'] = 'sunken'
        frame2['padding'] = 5 
        frame2.grid(column=1, row=0, pady=10)
        # #frame.grid(column=0,row=0, padx = 10, pady=10)
        frame2.pack_propagate(0)
        frame2.grid_propagate(0)

        current_var = StringVar()
        #combobox = CTkComboBox(frame2, textvariable=current_var)
        '''self.combobox = ttk.Combobox(frame2, values = ['Video Test', 'Calibration', 'Main Pipeline','fooltest'], state='readonly')
        #combobox['values'] = ('Video Test', 'Calibration', 'Main Pipeline')
        #combobox['state'] = 'readonly'
        self.combobox.set('Video Test')
        #combobox.current(0)
        self.combobox.grid(row=1,column=2,pady = 10)#pack(side='top', pady = 10)
        self.combobox.bind("<<ComboboxSelected>>",lambda e: frame2.focus())'''

        self.text = Text(frame2, height=15, width=30, bg = 'black', fg = 'green2', padx=5, pady=5, yscrollcommand=True,wrap='word')
        self.text.place(x= 5, y = 200)#pack(side = 'bottom')
        self.text.insert('1.0', 'Welcome!')
        self.text['state'] = 'disabled'

        self.frame3 = ttk.Frame(root, height = 960, width= 1280)
        self.frame3['borderwidth'] = 2
        self.frame3['relief'] = 'sunken'
        #frame3['padding'] = 5 
        self.frame3.grid(column = 0, row = 0, rowspan= 2)
        self.frame3.grid_propagate(0)
        self.can = Canvas(self.frame3, height = 960, width = 1280)
        self.can.grid(row = 0, column = 0)
        #self.can.config(bg= 'blue')
        


        ######################################################## All buttons go here ###################
        '''self.execute_button = Button(frame2, text="Execute", command = self.execute)
        self.execute_button.grid(row=2,column=2)#pack(side='top')'''
        #self.root.bind("<Escape>",lambda x: self.root.destroy())

        exit_button = Button(frame2, text="Exit",height=1,width=10, command=self.closeall)
        exit_button.place(x = 130, y = 95)#pack(side='bottom', pady = 10)

        enable_button = Button(frame2, text="Enable",height=1,width=10, command=Bot.enable)
        enable_button.place(x = 130, y = 15)

        disable_button = Button(frame2, text="Disable",height=1,width=10, command=Bot.disable)
        disable_button.place(x = 130, y = 45)

        ###### Run and calibrate buttons
        calibrate_button = Button(frame2, text="Calibrate",height=1,width=10, command=self.create_calib_buttons)
        calibrate_button.place(x = 10, y = 15)

        

        '''
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
        '''
        #self.root.mainloop()
    ########################################## class functions for switching between calibrate and run etc

    def create_calib_buttons(self):

        for widget in self.frame.winfo_children():
            widget.destroy()

        mananchor_button = Button(self.frame, text="Get petri dish anchors", command=self.get_anchors)
        mananchor_button.place(x=15,y=15)

        caldish_button = Button(self.frame, text="Calibrate petri dish", command=self.initialise_calibrate_dish)
        caldish_button.place(x=15,y=45)

        calplate_button = Button(self.frame, text="Get well plate corners", command=self.get_well_plate_anchors)
        calplate_button.place(x=15,y=75)
        
        testplate_button = Button(self.frame, text="Test well plate", command=self.test_96_wellplate)
        testplate_button.place(x=15,y=105)



    def create_finish_calibration_buttons(self):  # this is for petri dish anchors only
        for widget in self.frame.winfo_children():
            widget.destroy()

        Finish_button = Button(self.frame, text="Finish",height=1,width=10, command=self.finalise_calibration)
        Finish_button.place(x = 10, y = 15)

        Abort_button = Button(self.frame, text="Abort",height=1,width=10, command=self.abort)
        Abort_button.place(x = 130, y = 15)

    ############################################# Functions that actually do the work

    def test_96_wellplate(self):
        grid = np.load('well_plate_96_tk.npy')
        Bot.enable()
        iter = 16
        for coord in grid:
            if iter < 1:
                break
            iter = iter-1
            x,y = coord
            Bot.move.MovL(x,y,-40,0)
            Bot.move.Sync()
            Bot.move.MovL(x,y,-60,0)
            Bot.move.Sync()
            Bot.move.MovL(x,y,-40,0)
            Bot.move.Sync()
        Bot.disable()


    def get_well_plate_anchors(self):

        self.ins_message('Manually position the robot above the four corner cells in the well plate, press s to record, when done, press esc')
        self.root.update()

        keys = utils.Keyboard(Bot.dash) # initializing the class Keyboard from the module (file that houses functions (methods) and classes utils and we are passing the parameter dash
        # we are giving the class Keyboard a connection to the robot which is called dash, dash is an object of the class dashboard
        keys.execute() # Keyboard has a method called execute, use this to record the position of the robot by pressing s. if finished press esc
        # we want to find corners of well plate so we record the position of the 4 corners and use this information later, just one use of execute of class keyboard
        # this cell calculates the grid for the 96 well plate
        print ('length is', len(keys.coords))
        if len(keys.coords) == 4:
            well_plate = utils.assign_corners(keys.coords, reverse=True) # assign_corners is a method in the module utils 
            left_side_points = np.linspace(well_plate['ul'], well_plate['ll'], 12)[:,:2]
            right_side_points = np.linspace(well_plate['ur'], well_plate['lr'], 12)[:,:2]
            grid = []
            for i in range(len(left_side_points)):
                x1, y1 = left_side_points[i]
                x2, y2 = right_side_points[i]
                a = (x2-x1)/(y2-y1)
                b = x1 - a*y1
                ys = np.linspace(y1,y2,8) 
                xs = a*ys + b
                grid += (list(zip(xs,ys)))

            np.save('well_plate_96_tk.npy',np.array(grid)) # saves well plate grid into a file named 'well_plate_96_tk.npy', we want to save this so we can use later and not repeat process
            #np.load('well_plate_96_tk.npy')
            self.ins_message('Well plate saved!')
        else:
            self.ins_message('Ensure you have four coordinates!')



    def get_anchors(self):

        self.ins_message('Arrow keys: move, s key: save position, when done, press esc')
        self.root.update()
        
        Bot.enable()
        # use this to find points (fix z at -38)
        Bot.move.MovJ(250, -20, -38, 0)
        Bot.move.Sync()
        manmove = utils.ManualMove(Bot.move, Bot.dash) # used to control robot with keyboard, uses key presses to control robot 
        manmove.execute()
        # after above step, display this
        manmove.coords

        self.ins_message('saved ' + str(len(manmove.coords)) + ' coordinates')
        #save it   
        np.save('anchors_tk.npy', manmove.coords)


    def initialise_calibrate_dish(self):

        Bot.enable()

        self.recorded = []
        self.anchorflag = 0

        for widget in self.frame.winfo_children():
            widget.destroy()

        Next_button = Button(self.frame, text="Execute",height=1,width=10, command=self.calibrate_dish)
        Next_button.place(x = 10, y = 15)

        Abort_button = Button(self.frame, text="Abort",height=1,width=10, command=self.abort)
        Abort_button.place(x = 130, y = 15)

    def abort(self):
        Bot.disable()
        self.can.delete('all')
        self.create_calib_buttons()
        self.ins_message('Petri Dish Calibration Aborted')

    def calibrate_dish(self):
        
        #anchors = keys.coords
        self.anchors = np.load('anchors_tk.npy')

        # anchors = [np.array([307.315193, -13.865066, -81,  -3.484534]),
        #  np.array([316.442224,  49.591866, -81,  -3.484533]),
        #  np.array([268.040607,  48.172406, -81,  -3.484533]),
        #  np.array([260.923249, -14.987419, -81,  -3.484531])]

        # anchor positions, positions of the laser that the camera recognizes to create a transformation matrix
        # allows you to transform pixel coordinates of an object to actual robot coordinates 

        # computer vision stuff

        cameraMatrix = np.load('./cam_matrices/cam_mtx.npy')
        dist = np.load('./cam_matrices/dist.npy')
        newCameraMatrix = np.load('./cam_matrices/newcam_mtx.npy')
        template = cv2.imread('template.png', 0)
        w, h = template.shape[::-1]

        # cap = cv2.VideoCapture(0)
        # cap = cv_core.set_res(cap, cv_corfind_contourse.camera_res_dict['1200'])

        # cv2.namedWindow('frame',  cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame', 1348, 1011)



        anchor = self.anchors[self.anchorflag]
        
        x,y,z,r = anchor
        Bot.move.MovL(x,y,z,r)
        Bot.move.Sync()

        #while(True):
        cap = cv2.VideoCapture(0)
        cap = cv_core.set_res(cap, cv_core.camera_res_dict['1200'])

        #cv2.namedWindow('frame',  cv2.WINDOW_NORMAL) # creating a GUI window cv2 is a module called open cv which has all the methods related to computer vision
        #cv2.resizeWindow('frame', 1348, 1011)
        
        ret, frame = cap.read() # how to access camera information, gives single frame that has been captured by the camera at the time of execution
        # gives ret which is a boolean (true/false) true if frame captured, frame gives a numpy array that is basically the image (if color, 3 channel array rgb)
        
        
        frame = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix) # need these 3 parameters to undistort the frame and give a new undistorted frame
        
        
        
        plot_img = frame.copy() # create a copy of the variable frame (create a copy of the image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # turn frame to grayscale 
        (minVal, maxVal, minLoc, maxLoc)=cv2.minMaxLoc(gray) # gives location of minimum and maximum pixel value, gives coordinates in pixels
        a, b = maxLoc # unpack max location into 2 variables, the x and y (a and b) in pixels
        
        top_left = (a-w, b-h) # make white rectangle around desired maximum pixel values
        bottom_right = (a+w, b+h)

        mask = np.zeros_like(frame) # brightest part not always the center point, we want to isolate the bright spot, we apply a mask onto the image 
        # we take a grayscale image and we apply a mask around the bright spot, we create absolute black image 
        cv2.rectangle(mask,top_left, bottom_right, (255,255,255), -1) # we create filled white rectangle around the bright spot 
        cv2.rectangle(plot_img,top_left, bottom_right, (255,255,255), 2) # puts white hollow rectangle onto the visual image 
        result = cv2.bitwise_and(frame.astype('uint8'), mask.astype('uint8')) # we multiply the frame by the mask which leaves only the bright spot
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) # result given in rgb and we want to go to grayscale, gives one grayscale array instead of 3 BGR
        ret, thresh = cv2.threshold(gray_result,240,255,cv2.THRESH_BINARY) # apply thresholding to take brightest desired pixels and ignore all other values
        M = cv2.moments(thresh) # this is what is finding the center by calculating the moments of the bright blob, calculates center of mass of a pixel blob
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(plot_img, (cX, cY), 5, (0, 255, 0), 2) # drawing a circle around the brightest blob for visual purposes 
            self.recorded.append((cX, cY)) # we record the x and y pixel values of the center of the blob so we can map the location onto the robots coordinates, at that point
        else:
            self.anchorflag = self.anchorflag - 1
            self.ins_message('Too dark!')

        #time.sleep(1)
        #cv2.imwrite('temp.ppm', plot_img)

 
        photo_r = PIL.Image.fromarray(plot_img)
        photo_hc = photo_r.resize((1280,960))
        self.photo = PIL.ImageTk.PhotoImage(image = photo_hc)
        self.can.create_image(0, 0, image = self.photo, anchor = NW)

        #cv2.imshow('frame',plot_img) # shows the image
        #cv2.imwrite(f'anchor_{idx}.jpg', plot_img)
        #cv2.waitKey(0) # index zero wait key, tells the computer to wait for any key press before continuing execution
        cap.release() # releases the camera from control of the computer, disconnects the camera from the computer and empties memory
        
        print(self.recorded)
        print(self.anchorflag)

        #cv2.destroyAllWindows() # when 4 loop finishes it destroys (closes) the graphical window 
        length = len(self.anchors)
        
        if self.anchorflag >= length-1:
            self.create_finish_calibration_buttons()

        self.anchorflag = self.anchorflag + 1


    def finalise_calibration(self):
        xys = [(arr[0], arr[1]) for arr in self.anchors]
        robot_coor = utils.assign_corners(xys, reverse=True) # assign corners to the robot coordinates at the 4 corner positions 
        pix_coor = utils.assign_corners(self.recorded) # assign corners to the pixel coordinates at the 4 corner positions 

        features_mm_to_pixels_dict = {} # setting up an empty dictionary to store the mapping of the corners from coordinate to pixel
        for key, value in robot_coor.items():
            features_mm_to_pixels_dict[value] = pix_coor[key]


        self.tf_mtx = cv_core.compute_tf_mtx(features_mm_to_pixels_dict) # method of cv_core module that calculates transformation matrix
        # takes the dictionary and solves the system of linear equations that gives the transformation matrix and gives the actual relation between the pixels and millimeters 
        self.create_calib_buttons()
        self.ins_message('Petri Dish Calibrated')
        self.can.delete('all')
        Bot.disable()

    def ins_message(self,text:str): # TEXT MUST BE STR
        self.text['state'] = 'normal'
        self.text.insert(END, '\n\n'+text)
        self.text['state'] = 'disabled'
        self.text.see('end')



    '''def execute(self):
        if not self.cap_on:
            value = self.combobox.get()
            self.execute_button.config(text='stop')
            if value == 'Video Test':
                message = 'Testing video feed...'
                self.start_vid(self.video_test)
            elif value == 'Calibration':
                message = 'Calibrating...'
            elif value == 'fooltest':
                message = 'fooltesting'
                self.movetest()
  
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
            self.execute_button.config(text='Execute')'''

    # def movetest(self):
    #     p = utils.get_pose(Bot.dash, verbose=False)

    #     Bot.move.MovL(p[0], p[1], (p[2]+20),  p[3])
    #     Bot.move.Sync()
    #     Bot.move.MovL(p[0], p[1], (p[2]-20),  p[3])
    #     Bot.move.Sync()
    #     Bot.move.MovL(p[0], p[1], p[2],  p[3])
    #     Bot.move.Sync()

    #     self.execute_button.config(text='Execute')


    '''def start_vid(self, func):
        self.stop_vid()
        self.cap_on = True
        self.cap = cv2.VideoCapture(0) 
        self.cap = cv_core.set_res(self.cap, cv_core.camera_res_dict['1200'])
        func()'''

    '''def stop_vid(self):
        self.cap_on = False
        
        if self.cap:
            self.cap.release()
        self.canvas.delete('all')'''

    '''def video_test(self):
        # Get a frame from the video source
        if self.cap_on:
            ret, frame = self.cap.read()

            if ret:
                frame = cv2.resize(frame, (1332,999), interpolation = cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = NW)

            self.root.after(10, self.video_test)'''

    def closeall(self):
        try:
            Bot.disable()
        except:
            None
        root.destroy()


    '''def main_pipe(self):
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

            self.root.after(10, self.main_pipe)'''



class bot():  # contains the global varuiables to control the robot

    def __init__(self):

        self.dash = DobotApiDashboard('192.168.1.6', 29999) # dash is the object that is connected to the robot and gets information from the dashboard
        self.move = DobotApiMove('192.168.1.6', 30003) # the object that allows you to control the movement of the robot
  
    def enable(self):
        self.dash.ClearError()
        self.dash.EnableRobot()

    def disable(self):
        self.dash.DisableRobot()





Bot = bot()
    


root = Tk()
RobotApp(root, "Tkinter and OpenCV",Bot)
root.mainloop()