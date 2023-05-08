from tkinter import *
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
from deps import cv_core
import numpy as np
from deps.dobot_api import DobotApiDashboard, DobotApiMove
from deps import utils
import time
import math
import collections


class RobotApp:

    '''self.tf_mtx saves the petri dish calibation matrix
    
    '''


    def __init__(self, root, window_title,Bot) -> None:
        self.root = root
        self.root.title(window_title)
        self.cap_on = False
        self.cap = None

        self.recorded = []
        self.anchorflag = 0
        self.autoflag = 0

        #self.root.geometry("1920x1080")


        self.frame = ttk.Frame(self.root, width=250, height=425)
        self.frame['borderwidth'] = 2
        self.frame['relief'] = 'sunken'
        self.frame['padding'] = 5 
        self.frame.grid(column=1, row=1, padx=10, pady=10)
        self.frame.pack_propagate(0)


        '''self.canvas = Canvas(frame, width = 1332, height = 999)
        self.canvas.pack()'''


        self.frame2 = ttk.Frame(self.root, width=250, height=425)
        #frame2 = CTkFrame(root, width=250, height=825)
        self.frame2['borderwidth'] = 2
        self.frame2['relief'] = 'sunken'
        self.frame2['padding'] = 5 
        self.frame2.grid(column=1, row=0, pady=10)
        # #frame.grid(column=0,row=0, padx = 10, pady=10)
        self.frame2.pack_propagate(0)
        self.frame2.grid_propagate(0)

        current_var = StringVar()
        #combobox = CTkComboBox(self.frame2, textvariable=current_var)
        '''self.combobox = ttk.Combobox(self.frame2, values = ['Video Test', 'Calibration', 'Main Pipeline','fooltest'], state='readonly')
        #combobox['values'] = ('Video Test', 'Calibration', 'Main Pipeline')
        #combobox['state'] = 'readonly'
        self.combobox.set('Video Test')
        #combobox.current(0)
        self.combobox.grid(row=1,column=2,pady = 10)#pack(side='top', pady = 10)
        self.combobox.bind("<<ComboboxSelected>>",lambda e: self.frame2.focus())'''

        self.text = Text(self.frame2, height=15, width=30, bg = 'white', fg = 'black', padx=5, pady=5, yscrollcommand=True,wrap='word')
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
        


        ######################################################## CREATES BUTTONS IN UPPER FRAME ###################
        '''self.execute_button = Button(self.frame2, text="Execute", command = self.execute)
        self.execute_button.grid(row=2,column=2)#pack(side='top')'''
        #self.root.bind("<Escape>",lambda x: self.root.destroy())

        exit_button = Button(self.frame2, text="Exit",height=1,width=10, command=self.closeall)
        exit_button.place(x = 130, y = 95)#pack(side='bottom', pady = 10)

        #enable_button = Button(self.frame2, text="Enable",height=1,width=10, command=Bot.enable)
        #enable_button.place(x = 130, y = 15)

        #disable_button = Button(self.frame2, text="Disable",height=1,width=10, command=Bot.disable)
        #disable_button.place(x = 130, y = 45)

        ###### Run and calibrate buttons
        calibrate_button = Button(self.frame2, text="Calibrate",height=1,width=10, command=self.create_calib_buttons)
        calibrate_button.place(x = 10, y = 15)

        run_button = Button(self.frame2, text="Run",height=1,width=10, command=self.create_run_buttons)
        run_button.place(x = 10, y = 45)
        
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
    ########################################## class functions for switching between buttons

    def create_calib_buttons(self):

        for widget in self.frame.winfo_children():
            widget.destroy()

        mananchor_button = Button(self.frame, text="Get petri dish anchors", command=self.get_anchors)
        mananchor_button.place(x=15,y=15)

        caldish_button = Button(self.frame, text="Calibrate petri dish", command=self.initialise_calibrate_dish)
        caldish_button.place(x=15,y=45)

        calplate_button = Button(self.frame, text="Get well plate corners", command=self.get_well_plate_anchors)
        calplate_button.place(x=15,y=105)

        water_button = Button(self.frame, text="Get water location", command=self.get_water_coordinates)
        water_button.place(x=15,y=135)
        
        testplate_button = Button(self.frame, text="Test well plate", command=self.test_96_wellplate)
        testplate_button.place(x=15,y=165)

        pipette_button = Button(self.frame, text="Verification/ positioning", command=self.cuboid_recognition_and_positioning)
        pipette_button.place(x=15,y=75)

    def create_finish_calibration_buttons(self):  # this is for petri dish anchors only
        for widget in self.frame.winfo_children():
            widget.destroy()

        Finish_button = Button(self.frame, text="Finish",height=1,width=10, command=self.finalise_calibration)
        Finish_button.place(x = 10, y = 15)

        Abort_button = Button(self.frame, text="Abort",height=1,width=10, command=self.abort)
        Abort_button.place(x = 130, y = 15)

    def create_run_buttons(self):

        for widget in self.frame.winfo_children():
            widget.destroy()

        run_button = Button(self.frame, text="Run Manually", command=self.run)
        run_button.place(x=15,y=45)

        autorun_button = Button(self.frame, text="Run automatically", command=self.autorun)
        autorun_button.place(x=15,y=75)

        self.input_start_picknplace = Text(self.frame, height = 1, width = 10)
        self.input_start_picknplace.place(x=15, y=15)

        input_label = Label(self.frame, text=":Starting cell")
        input_label.place(x=100, y=15)
        
    ############################################# Functions for calibration

    def test_96_wellplate(self):
        '''tests first two rows of well plate by positioning robot above each well'''

        self.del_msg()
        if self.cap_on == 1:
            return
        grid = np.load('well_plate_96_tk.npy')
        Bot.enable()
        iter = 96
        for coord in grid:
            if iter < 89:
                if iter%8 != 1:
                    iter = iter-1
                    continue
            iter = iter-1
            x,y = coord
            Bot.move.MovL(x,y,0,0)
            Bot.move.Sync()
            Bot.move.MovL(x,y,-35,0)
            Bot.move.Sync()
            Bot.move.MovL(x,y,0,0)
            Bot.move.Sync()
        #Bot.disable()

    def get_well_plate_anchors(self):
        '''Gets the coordinates of the well plate and saves the well plate to use in picking and placing'''

        self.del_msg()

        if self.cap_on == 1:
            return

        self.ins_message('Manually position the robot above the four corner cells in the well plate, press s to record, when done, press esc')
        self.root.update()

        keys = utils.Keyboard(Bot.dash) # initializing the class Keyboard from the module (file that houses functions (methods) and classes utils and we are passing the parameter dash
        # we are giving the class Keyboard a connection to the robot which is called dash, dash is an object of the class dashboard
        keys.execute() # Keyboard has a method called execute, use this to record the position of the robot by pressing s. if finished press esc
        # we want to find corners of well plate so we record the position of the 4 corners and use this information later, just one use of execute of class keyboard
        # this cell calculates the grid for the 96 well plate
        #print ('length is', len(keys.coords))
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
  
    def get_water_coordinates(self):
        '''gets coordinates of the water well'''

        self.del_msg()
        if self.cap_on == 1:
            return

        self.ins_message('Manually position the robot above the water well, press s to record, when done, press esc')
        self.root.update()

        keys = utils.Keyboard(Bot.dash)
        keys.execute()
        if len(keys.coords)>0:
            l = len(keys.coords) - 1
            np.save('water_coordinates_tk.npy',np.array(keys.coords[l]))
        else:
            self.ins_message('Please select a coordinate!')

    def get_anchors(self):
        '''gets anchors for calibrating the robot and saves them in a file'''

        self.del_msg()
        if self.cap_on == 1:
            return
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

        #Bot.disable()

    def initialise_calibrate_dish(self):
        '''moves robot to anchors gathered in the previous function to calibrate the robot'''

        self.del_msg()

        if self.cap_on == 1:
            return
        Bot.enable()

        self.recorded = []
        self.anchorflag = 0

        self.cameraMatrix = np.load('./cam_matrices/cam_mtx_1944.npy')
        self.dist = np.load('./cam_matrices/dist_1944.npy')
        self.newCameraMatrix = np.load('./cam_matrices/newcam_mtx_1944.npy')

        for widget in self.frame.winfo_children():
            widget.destroy()

        Next_button = Button(self.frame, text="Execute",height=1,width=10, command=self.calibrate_dish)
        Next_button.place(x = 10, y = 15)

        Abort_button = Button(self.frame, text="Abort",height=1,width=10, command=self.abort)
        Abort_button.place(x = 130, y = 15)

    def abort(self):
        '''used for robot calibration'''
        #Bot.disable()
        self.can.delete('all')
        self.create_calib_buttons()
        self.ins_message('Petri Dish Calibration Aborted')

    def calibrate_dish(self):
        '''used for robot calibration'''
        
        #anchors = keys.coords
        self.anchors = np.load('anchors_tk.npy')

        # anchors = [np.array([307.315193, -13.865066, -81,  -3.484534]),
        #  np.array([316.442224,  49.591866, -81,  -3.484533]),
        #  np.array([268.040607,  48.172406, -81,  -3.484533]),
        #  np.array([260.923249, -14.987419, -81,  -3.484531])]

        # anchor positions, positions of the laser that the camera recognizes to create a transformation matrix
        # allows you to transform pixel coordinates of an object to actual robot coordinates 

        # computer vision stuff

        template = cv2.imread('laser_template.png', 0)
        w, h = template.shape[::-1]
        #print(w,h)

        # cap = cv2.VideoCapture(0)
        # cap = cv_core.set_res(cap, cv_corfind_contourse.camera_res_dict['1944'])

        # cv2.namedWindow('frame',  cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame', 1348, 1011)



        anchor = self.anchors[self.anchorflag]
        
        x,y,z,r = anchor
        Bot.move.MovL(x,y,z,r)
        Bot.move.Sync()

        #while(True):
        cap = cv2.VideoCapture(0)
        cap = cv_core.set_res(cap, cv_core.camera_res_dict['1944'])

        #cv2.namedWindow('frame',  cv2.WINDOW_NORMAL) # creating a GUI window cv2 is a module called open cv which has all the methods related to computer vision
        #cv2.resizeWindow('frame', 1348, 1011)
        
        ret, frame = cap.read() # how to access camera information, gives single frame that has been captured by the camera at the time of execution
        # gives ret which is a boolean (true/false) true if frame captured, frame gives a numpy array that is basically the image (if color, 3 channel array rgb)
        
        frame = cv2.undistort(frame, self.cameraMatrix, self.dist, None, self.newCameraMatrix) # need these 3 parameters to undistort the frame and give a new undistorted frame
        plot_img = frame.copy() # create a copy of the variable frame (create a copy of the image)

        #res = cv2.matchTemplate(frame,template, cv2.TM_CCORR_NORMED)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # top_left = max_loc
        # bottom_right = (top_left[0] + w, top_left[1] + h)
        
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
        

        #cv2.destroyAllWindows() # when 4 loop finishes it destroys (closes) the graphical window 
        length = len(self.anchors)
        
        if self.anchorflag >= length-1:
            self.create_finish_calibration_buttons()

        self.anchorflag = self.anchorflag + 1

    def finalise_calibration(self):
        '''used for robot calibration'''

        xys = [(arr[0], arr[1]) for arr in self.anchors]
        robot_coor = utils.assign_corners(xys, reverse=True) # assign corners to the robot coordinates at the 4 corner positions 
        pix_coor = utils.assign_corners(self.recorded) # assign corners to the pixel coordinates at the 4 corner positions 

        features_mm_to_pixels_dict = {} # setting up an empty dictionary to store the mapping of the corners from coordinate to pixel
        for key, value in robot_coor.items():
            features_mm_to_pixels_dict[value] = pix_coor[key]


        self.tf_mtx = cv_core.compute_tf_mtx(features_mm_to_pixels_dict) # method of cv_core module that calculates transformation matrix
        # takes the dictionary and solves the system of linear equations that gives the transformation matrix and gives the actual relation between the pixels and millimeters
        np.save('tfm_mtx_tk.npy', self.tf_mtx)
        self.create_calib_buttons()
        self.ins_message('Petri Dish Calibrated')
        self.can.delete('all')
        #Bot.disable()

    '''the following functions display the cuboids on the screen as a video, and let you select one to move the robot to, so that you can position the pipette
    where the cuboid is'''

    def set_mouse_pos(self,event):
        '''records mouse left click'''
        self.x = int(2592*event.x/1280)
        self.y = int(1944*event.y/960)
        self.set_mouse_pos_r(event)
        #print(self.x,self.y)
        #self.anchorflag = 1
        #print('flag',self.anchorflag)

        select(self.cont,self.x,self.y,None,None)

    def end(self):
        '''runs at the end of the cuboid selection'''

        self.can.unbind('<Double-1>')
        self.can.unbind('<Button-3>')
        self.root.unbind('<Button-4>')
        self.root.unbind('<Button-5>')
        self.root.unbind('<Return>')
        self.root.unbind('<l>')
        self.root.unbind('<i>')
        self.root.unbind('<q>')
        self.root.unbind('<Left>')
        self.root.unbind('<Right>')
        self.root.unbind('<Up>')
        self.root.unbind('<Down>')
        self.root.unbind('<w>')
        self.root.unbind('<a>')
        self.root.unbind('<s>')
        self.root.unbind('<d>')
        self.root.unbind('<e>')
        self.root.unbind('<c>')
        self.cap.release()
        self.can.delete('all')
        self.cap_on = 0

        self.tinycan.destroy()
        self.create_calib_buttons()

        for widget in self.frame2.winfo_children():
            widget["state"] = "normal"
        self.text['state'] = 'disabled'

        #print('entered')

    def move_to_cuboid(self):
        '''moves bot to the selected cuboid'''

        Bot.enable()
        M = cv2.moments(self.cont.selected[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        self.tf_mtx = np.load('tfm_mtx_tk.npy')
        X, Y, _ = self.tf_mtx @ (cX, cY, 1)
        Bot.move.MovL(X, Y, -38, 0)
        Bot.move.Sync()
    
    def selectcuboid(self,_):
        '''handles all functions when decision is made to select the cuboid'''

        if self.cont.selected == []:
            self.ins_message('Select cuboid!')
            return
        self.move_to_cuboid()
        self.end()

    def dont_selectcuboid(self,_):
        '''end the function without selecting cuboid'''

        self.end()

    def set_mouse_pos_r(self,event):
        '''sets the mouse coordinates in case of a right click'''
        self.rx = int(2592*event.x/1280)
        self.ry = int(1944*event.y/960)
        self.set_tinyscope()

    def set_tinyscope(self):
        '''sets zoom and scope on the tiny screen, used for seeing cuboids close up'''
        if self.rx >=self.scopewidth: 
            self.scopexm = self.rx - self.scopewidth
        else:
            self.scopexm = 0

        if self.rx < 2591 - self.scopewidth: 
            self.scopexp = self.rx + self.scopewidth
        else:
            self.scopexp = 2591

        if self.ry >=self.scopewidth: 
            self.scopeym = self.ry - self.scopewidth
        else:
            self.scopeym = 0

        if self.ry < 1943 - self.scopewidth: 
            self.scopeyp = self.ry + self.scopewidth
        else:
            self.scopeyp = 1943

    def pressed_4(self,_):
        if self.scopewidth < 200:
            self.scopewidth = self.scopewidth + 5
        self.set_tinyscope()

    def pressed_5(self,_):
        if self.scopewidth > 5:
            self.scopewidth = self.scopewidth - 5
        self.set_tinyscope()

    def reset_scope_width(self,_):
        self.scopewidth = 50
        self.set_tinyscope()

    def leftKey(self,_):
        if self.rx >0:
            self.rx = self.rx-1
        #print(self.rx)
        self.set_tinyscope()

    def rightKey(self,_):
        if self.rx < 1520:
            self.rx = self.rx+1
        #print(self.rx)
        self.set_tinyscope()

    def upKey(self,_):
        if self.ry >0:
            self.ry = self.ry-1
        #print(self.ry)
        self.set_tinyscope()

    def downKey(self,_):
        if self.ry < 1220:
            self.ry = self.ry+1
        #print(self.ry)
        self.set_tinyscope()

    def wKey(self,_):
        if self.cont.best_circ[1] >0:
            self.cont.best_circ[1] = self.cont.best_circ[1]-1
            self.cont.big_circ[1] = self.cont.big_circ[1]-1
        
    def sKey(self,_):
        if self.cont.best_circ[1] < 1944:
            self.cont.best_circ[1] = self.cont.best_circ[1] + 1
            self.cont.big_circ[1] = self.cont.big_circ[1] + 1

    def aKey(self,_):
        if self.cont.best_circ[0] >0:
            self.cont.best_circ[0] = self.cont.best_circ[0]-1
            self.cont.big_circ[0] = self.cont.big_circ[0]-1

    def dKey(self,_):
        if self.cont.best_circ[0] < 2592:
            self.cont.best_circ[0] = self.cont.best_circ[0] + 1
            self.cont.big_circ[0] = self.cont.big_circ[0] + 1

    def eKey(self,_):
        if self.cont.best_circ[2] < 972:
            self.cont.best_circ[2] = self.cont.best_circ[2] + 1

    def cKey(self,_):
        if self.cont.best_circ[2] > 200:
            self.cont.best_circ[2] = self.cont.best_circ[2] - 1

    def cuboid_recognition_and_positioning(self):
        '''the actual function that starts the recognition and positioning'''
        if self.cap_on == 1:
            return
        
        self.del_msg()
        for widget in self.frame.winfo_children():
            widget.destroy()
        for widget in self.frame2.winfo_children():
            widget["state"] = "disabled"

        self.tinycan = Canvas(self.frame, height = 230, width = 230)#,bg='black')
        self.tinycan.place(x=0,y=0)

        Bot.enable()
        Bot.move.JointMovJ(-30,0,0,0)
        Bot.move.Sync()

        self.can.bind('<Double-1>',self.set_mouse_pos)
        self.can.bind('<Button-3>',self.set_mouse_pos_r)
        self.root.bind('<Button-4>',self.pressed_4)
        self.root.bind('<Button-5>',self.pressed_5)
        
        self.root.bind('<Return>',self.selectcuboid)
        self.root.bind('<i>',self.reset_scope_width)
        self.root.bind('<q>',self.dont_selectcuboid)

        self.root.bind('<Left>', self.leftKey)
        self.root.bind('<Right>', self.rightKey)
        self.root.bind('<Up>', self.upKey)
        self.root.bind('<Down>', self.downKey)

        self.root.bind('<w>', self.wKey)
        self.root.bind('<a>', self.aKey)
        self.root.bind('<s>', self.sKey)
        self.root.bind('<d>', self.dKey)
        self.root.bind('<e>', self.eKey)
        self.root.bind('<c>', self.cKey)

        self.cap_on = 1

        self.cont = cv_core.Contours() # define class of methods for cuboid detection, initialize class
        #tf_mtx = np.load('tfm_mtx_tk.npy')
        def on_change(val): pass

        self.anchorflag = 0

        self.offset = 40 # create smaller inner circle in petri dish to locate cuboids 
        self.cap = cv2.VideoCapture(0) # gets access to the camera 
        self.cap = cv_core.set_res(self.cap, cv_core.camera_res_dict['1944']) # sets the resolution of the camera to 1944 x 1600

        self.cameraMatrix = np.load('./cam_matrices/cam_mtx_1944.npy')
        self.dist = np.load('./cam_matrices/dist_1944.npy')
        self.newCameraMatrix = np.load('./cam_matrices/newcam_mtx_1944.npy')
        '''
        cv2.namedWindow('frame',  cv2.WINDOW_NORMAL) # create window
        cv2.resizeWindow('frame', 1348, 1011) # set resolution of window
        cv2.createTrackbar('Manual Lock', 'frame', 0, 1, on_change) # create trackbar between values of 0 and 1, basically a switch (manual lock of the circle 
        #recognition)
        # stops trying to recognize the petri dish (locked) increases performance of more fps because it recognizes a lot of circles which takes time so this locks it 
        # circle recognition needs optimizing 
        cv2.createTrackbar('Mask Offset', 'frame', offset, 150, on_change) # second trackbar allows control of offset which changes the size of the petri dish detection circle
        cv2.setMouseCallback('frame', cont.mousecallback) # set a callback function for double clicks of the mouse, this allows us to select cuboids by double clicking
        # initializes a double click response to select a certain contour on the screen 
        '''
        
        self.x = 0
        self.y = 0
        self.rx = 800
        self.ry = 600
        self.scopewidth = 50
        self.set_tinyscope()
        self.cont.locked = True
        self.idx = 0 
        self.prev_point = (0,0,0)

        self.ins_message('double click and enter: move bot to highlighted cuboid\nRight click: zoom in in small window\ni: reset zoom\nl: toggle lock\narrow keys: pan zoomed view\nw,a,s,d: pan dish location\ne,c: zoom dish in/out\nq: quit')

        #while(True): # we want a video stream so we want a while loop to continuously take new images until loop is broken
        self.cuboid_recnpos_vid()          

    def cuboid_recnpos_vid(self):
                
        if self.cap_on == 1:
            offset = self.offset

            # if self.anchorflag == 1:
            #     select(self.cont,self.x,self.y,None,None)
            #     self.anchorflag = 0

            ret, frame = self.cap.read()

            frame = cv2.undistort(frame, self.cameraMatrix, self.dist, None, self.newCameraMatrix) # need these 3 parameters to undistort the frame and give a new undistorted frame
            
            #frame = cv2.undistort(frame, self.cameraMatrix, self.dist, None, self.newCameraMatrix)
            plot_img = frame#.copy() # create a copy of the frame so things can be drawn on it without altering original image


            ######### Interlude: for tinycan:
            tinypic = plot_img[self.scopeym:self.scopeyp,self.scopexm:self.scopexp]
            tinyphoto_r = PIL.Image.fromarray(tinypic[:,:,::-1])
            tinyphoto_hc = tinyphoto_r.resize((260,260),resample=0)
            self.tinyphoto = PIL.ImageTk.PhotoImage(image = tinyphoto_hc)
            self.tinycan.delete('all')
            self.tinycan.create_image(0, 0, image = self.tinyphoto, anchor = NW)

            # print(tinypic.size, self.scopexp)
            # print('x and y', self.rx,self.ry)
            # print(self.scopexm,self.scopexp,self.scopeym,self.scopeyp)
            


            #if self.val == 1:
            #    val = 1#cv2.getTrackbarPos('Manual Lock', 'frame') # checks position of trackbar for manual lock, if trackbar 1, then val = 1, this turns off circle detection
            #else:
            #    val = 0
            #offset = cv2.getTrackbarPos('Mask Offset', 'frame') # the trackbar with the offset, changes the offset value for the circle

            # if cont.best_circ is None:
            #     while cont.best_circ is None:
            #         pt = cont.find_contours(frame, 10, offset)
            # else:
            pt = self.cont.find_contours(frame, 10, offset) # 10 og,takes the frame and looks for the circle of the petri dish, also looks for the cuboids 
                
            a,b,r = pt # gives the x y coordinates and the radius of the circle
            
            cv2.circle(plot_img, (a, b), r, (3, 162, 255), 2) # circles being drawn on image to locate petri dish
            #cv2.circle(plot_img, (a, b), 1, (0, 0, 255), 3)
            cv2.circle(plot_img, (a, b), r - offset, (0, 255, 0), 2)
            # print(cont.big_circ[2])
            cv2.circle(plot_img, self.cont.big_circ[:2], int(self.cont.big_circ[2]), (0, 0, 255), 2)
            cv2.putText(plot_img, f"{int(self.cont.big_circ[2]*2)}px = 60mm", (int(self.cont.big_circ[0]-25), int(self.cont.big_circ[1]) - int(self.cont.big_circ[2]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2) # how to decide size of cuboids, use diameter of petri dish being 60 mm to approximate size of cuboids
                # just a text sign around the biggest circle of the image (petri dish) saying that this circle is 60 mm
    
            cv2.drawContours(plot_img, self.cont.singular, -1,(0,255, 0),2) # two lists of contours (cuboid contours) that we draw on the plotting image
            cv2.drawContours(plot_img, self.cont.clusters, -1,(0,0,255),2) # we do this to distinguish between clusters and cuboids 

            if self.cont.selected: # if we selected cuboids with the double click, we draw them in blue
                cv2.drawContours(plot_img, self.cont.selected, -1,(255,0,0),2) # BGR, this is blue contour 

            for c in self.cont.singular: # we find the centers of the contours similar to how we found centers with the calibration step
                # compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"]) # the function moments gives a mysterious dictionary with elements of matrix, it seems that you gotta divide one element by another to get x&y
                cY = int(M["m01"] / M["m00"])
                
                cv2.circle(plot_img, (cX, cY), 2, (0, 0, 255), -1) # just draw a cirlce around the contour 
                cv2.putText(plot_img, f"{cX},{cY}", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(plot_img, f"{cv2.contourArea(c)}", (cX - 20, cY - 20), # putting the area of the cuboids onto the screen as text next to the cuboid
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            #out.write(with_contours)

            for c in self.cont.clusters: # same thing for clustes as above but no printed text for cuboid area or coordinate
                # compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                #cv2.drawContours(plot_img, [c], -1, (0, 255, 0), 2)
                cv2.circle(plot_img, (cX, cY), 2, (0, 0, 255), -1)
                # cv2.putText(plot_img, f"{cX},{cY}", (cX - 20, cY - 20),
                # cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                # cv2.putText(plot_img, f"{cv2.contourArea(c)}", (cX - 20, cY - 20),
                # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)


            if self.prev_point is pt:
                self.idx += 1
            else:
                self.idx = 0
            self.prev_point = pt


            '''if self.val == 1: # if circle stays same for awhile then it will show as locked (30 frames the same)
                message = 'LOCKED'
                color = (0,255,0)
            else:
                message = 'SEARCHING' # more text messages 
                color = (0,0,255)
            '''

            cv2.putText(plot_img, "TARGET:", (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2) # text messages, target displays constantly
            #cv2.putText(plot_img, f"{message}", (125,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2) # displays message locked or searching
            cv2.putText(plot_img, f"Found: {len(self.cont.singular)}", (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2) # how many cuboids it sees
            
                # if cv2.waitKey(1) & 0xFF == ord('q'): # if you press q, the while loop will break and the video and locating will stop, then the camera is released and windows destroyed
                #     break

            #cap.release()
            #cv2.destroyAllWindows()
            #self.can.delete('all')
            #self.can.unbind('<Double-1>')
            #print('Run')

            photo_r = PIL.Image.fromarray(plot_img[:,:,::-1])
            photo_hc = photo_r.resize((1280,960))
            self.photo = PIL.ImageTk.PhotoImage(image = photo_hc)
            self.can.delete('all')
            self.can.create_image(0, 0, image = self.photo, anchor = NW)
            
            #self.cont = cont
            self.root.after(10,self.cuboid_recnpos_vid)

    ################################################################ Functions for picking and placing

    def calc_centers(self,conts):
        centers = []
        for i in range(len(conts)):
            M = cv2.moments(conts[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            X, Y, _ = self.tf_mtx @ (cX, cY, 1)
            centers.append([X,Y])
        return centers  

    def mindis(self,centers):
        dis = collections.defaultdict(lambda: np.inf)
        for i in range(len(centers)):
            x,y = centers[i]
            #print(x,y)
            for j in range(i+1,len(centers)):
                x1,y1 = centers[j]
                distance = ((x1-x)**2) + ((y1-y)**2)
                dis[i] = min(dis[i], distance)
                dis[j] = min(dis[j],distance)
        #return [(-v,k) for k,v in sorted(dis.items(), key = lambda item : item[0], reverse = True)]        
        dist = list(sorted(dis.items(), key = lambda item : item[1], reverse = True))
        val = 1.8
        return list(filter(lambda x: x[1] >= val, dist))  

    def abort_picknplace(self,_):
        ''''''
        if self.autoflag == 0:
            self.root.unbind('<Return>')
            self.root.unbind('<q>')
            self.root.unbind('<r>')

        for widget in self.frame2.winfo_children():
            widget["state"] = "normal"

        for widget in self.frame.winfo_children():
            widget["state"] = "normal"

        self.text['state'] = 'disabled'

        self.ins_message('end')

        self.can.delete('all')
        self.cap_on = 0
        self.cap.release()
        self.autoflag = 0

    def picknplace(self,_):
        ''''''
        #print(len(cont.singular))
        a = len(self.cont.singular)
        if a >= 1:
            # center = cont.contour_centers(cont.singular)[0]            
            #center = random.choice(cont.contour_centers(cont.singular))
            #X, Y, _ = tf_mtx @ (center[0], center[1], 1)
            #-38
            if a > 1:
                X, Y = self.cs[self.ds[0][0]]
            else:
                X, Y = self.cs[0]
            #print ('TROUBLE:',X,Y)
            Bot.move.MovL(X, Y, self.calibration_z + self.z_offset, 0)
            Bot.move.Sync()
            utils.correct_J4_angle(0, Bot.dash, Bot.move)
            # utils.correct_J4_angle(-360, dash, move)
            # utils.correct_J4_angle(0, dash, move)
            Bot.move.RelMovL(0,0, -self.z_offset) #(0,0,-36)
            Bot.move.Sync()
            utils.correct_J4_angle(120, Bot.dash, Bot.move)
            Bot.move.RelMovL(0,0,self.z_offset) #(0,0,36) #24 works!!
            Bot.move.Sync()
            x,y = self.grid[self.idx]
            self.idx += 1
            Bot.move.MovL(x,y,self.calibration_z + self.z_offset, 120)
            Bot.move.Sync()
            utils.correct_J4_angle(120, Bot.dash, Bot.move)
            Bot.move.RelMovL(0,0,-28) #(0,0,-27)
            Bot.move.Sync()
            utils.correct_J4_angle(-100, Bot.dash, Bot.move)
            Bot.move.RelMovL(0,0, 28) #(0,0,-27)
            Bot.move.Sync()

            #if (self.idx + 1)%9 == 0:
            #    ''''''
            #    self.water_run()


            self.run_loop()
        else:
            self.abort_picknplace(5)

    def water_run(self):

        Bot.move.MovL(self.water_dish[0], self.water_dish[1], self.water_dish[2]+35, self.water_dish[3])
        Bot.move.Sync()
        Bot.move.RelMovL(0,0,-35)
        Bot.move.Sync()
        utils.correct_J4_angle(480, Bot.dash, Bot.move)
        Bot.move.Sync()
        Bot.move.RelMovL(0,0,35)
        Bot.move.Sync()
        Bot.move.MovL(self.center_petri[0],self.center_petri[1],self.center_petri[2],self.center_petri[3])
        Bot.move.Sync()
        utils.correct_J4_angle(-480, Bot.dash, Bot.move)
        Bot.move.Sync()
        Bot.move.JointMovJ(-30,0,0,0)
        Bot.move.Sync()
        #Bot.move.movL(x,y,z,r) # move back to where??
        #Bot.move.Sync()

    def runagain(self,_):
        self.run_loop()

    def run(self):
        self.del_msg()
        Bot.enable()
        self.grid = np.load('well_plate_96_tk.npy')
        Bot.move.JointMovJ(-30,0,0,0)
        self.tf_mtx = np.load('tfm_mtx_tk.npy')
        self.anchors = np.load('anchors_tk.npy')
        self.water_dish = np.load('water_coordinates_tk.npy')

        self.cameraMatrix = np.load('./cam_matrices/cam_mtx.npy')
        self.dist = np.load('./cam_matrices/dist.npy')
        self.newCameraMatrix = np.load('./cam_matrices/newcam_mtx.npy')
        
        self.ctrs = []
        idx = self.input_start_picknplace.get("1.0", END)
        if type(idx) == int and idx > 0:
            self.idx = idx - 1
        else:
            self.idx = 0

        self.center_petri = np.array([0, 0, 0, 0])
        for i,anchors in enumerate(self.anchors):
            self.center_petri = self.center_petri + anchors
        self.center_petri = self.center_petri/(i+1)

        try:
            self.cont.locked = True
        except:
            self.cont = cv_core.Contours() # define class of methods for cuboid detection, initialize class
            self.cont.locked = True
        self.offset = 60 # create smaller inner circle in petri dish to locate cuboids 
        self.cap = cv2.VideoCapture(0) # gets access to the camera 
        self.cap = cv_core.set_res(self.cap, cv_core.camera_res_dict['1944']) # sets the resolution of the camera to 1944 x 1600

        self.cameraMatrix = np.load('./cam_matrices/cam_mtx_1944.npy')
        self.dist = np.load('./cam_matrices/dist_1944.npy')
        self.newCameraMatrix = np.load('./cam_matrices/newcam_mtx_1944.npy')

        if self.autoflag == 0:
            self.root.bind('<Return>',self.picknplace)
            self.root.bind('<q>',self.abort_picknplace)
            self.root.bind('<r>',self.runagain)

        for widget in self.frame2.winfo_children():
            widget["state"] = "disabled"
        
        for widget in self.frame.winfo_children():
            widget["state"] = "disabled"

        if self.autoflag == 0:
            self.ins_message('Enter: Pick and place highlighted cuboid\nr: refresh frame\nq: quit')
        else:
            self.ins_message('Running...')

        self.calibration_z = np.mean(self.anchors[:,2])
        self.z_offset = 35

        self.cap_on = 1
        self.run_loop()

    def run_loop(self):    
        
        if self.cap_on == 1:
            
            for i in range(4):   # TODO: Test with this and see if you still need five presses on the spacebar # Sarmad Hassan
                self.cap.read()    
            ret, frame = self.cap.read()
            
            
            #ret, frame = cap.read() 
            frame = cv2.undistort(frame, self.cameraMatrix, self.dist, None, self.newCameraMatrix) # need these 3 parameters to undistort the frame and give a new undistorted frame
            #frame = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)

            pt = self.cont.find_contours(frame, 10, self.offset) # takes the frame and looks for the circle of the petri dish, also looks for the cuboids 
                
            a,b,r = pt # gives the x y coordinates and the radius of the circle
            plot_img = frame.copy() # create a copy of the frame so things can be drawn on it without altering original image

            cv2.circle(plot_img, (a, b), r, (3, 162, 255), 2) # circles being drawn on image to locate petri dish
            cv2.circle(plot_img, (a, b), 1, (0, 0, 255), 3)
            cv2.circle(plot_img, (a, b), r - self.offset, (0, 255, 0), 2) 

            cv2.drawContours(plot_img, self.cont.singular, -1,(0,255, 0),2) # two lists of contours (cuboid contours) that we draw on the plotting image
            cv2.drawContours(plot_img, self.cont.clusters, -1,(0,0,255),2) # we do this to distinguish between clusters and cuboids

            cv2.putText(plot_img, f"Found: {len(self.cont.singular)}", (25,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2) # how many cuboids it sees
            self.cs = self.calc_centers(self.cont.singular)
            self.ds = np.array(self.mindis(self.cs))
            #print(len(cont.singular))
            #print(type(self.ds))
            #print(cont.singular[ds[0][0]])
            if self.ds.ndim > 1:
                cv2.drawContours(plot_img, [self.cont.singular[int(self.ds[0][0])]],-1,(255,0,0),2)
            elif self.ds.ndim == 1:
                if self.ds.size > 0:
                    cv2.drawContours(plot_img, [self.cont.singular[0]],-1,(255,0,0),2)
            
            #cv2.imshow('frame',plot_img) # this just displays the image 
            photo_r = PIL.Image.fromarray(plot_img[:,:,::-1])
            photo_hc = photo_r.resize((1280,960))
            self.photo = PIL.ImageTk.PhotoImage(image = photo_hc)
            self.can.delete('all')
            self.can.create_image(0, 0, image = self.photo, anchor = NW)
            
            
            #self.root.after(10,self.run_loop)
  ##
    def autorun(self):
        self.autoflag = 1
        self.run()
        self.root.update()
        self.picknplace(5)

        iter = 5

        while iter > 1 and self.autoflag == 1:
            self.run_loop()
            self.root.update()
            self.picknplace(5)
            iter = iter - 1

            if len(self.cont.singular) < 1:
                break
        if iter < 1:
            self.abort_picknplace(5)
        
    ######################### 

    def ins_message(self,text:str): # TEXT MUST BE STR
        self.text['state'] = 'normal'
        self.text.insert(END, '\n\n'+text)
        self.text['state'] = 'disabled'
        self.text.see('end')

    def del_msg(self):
        ''''''
        self.text['state'] = 'normal'
        self.text.delete("1.0","end")
        self.text['state'] = 'disabled'

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
        self.cap = cv_core.set_res(self.cap, cv_core.camera_res_dict['1944'])
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




try:
    Bot = bot()
    flag = 1
except:
    flag = 0

def select(cont,x,y,flags,param):
        """OpenCV mouse callback function for registering double clicks.
        In our case used to select individual cuboids and pick them insted of
        picking everything automatically. If there is another double click then
        deselect the cuboid.

        Args:
            event (int?): OpenCV event code. Usually an int?
            x (int?): x position of the click in the image.
            y (int?): y position of the click in the image.
            flags (_type_): No idea.
            param (_type_): Optional parameter the can be returned?
        """        

        if cont.selected:
            for contour in cont.selected:
                r=cv2.pointPolygonTest(contour, (x,y), False)
                if r > 0:
                    cont.selected.remove(contour)
                    return

        for contour in cont.singular:
            r=cv2.pointPolygonTest(contour, (x,y), False)
            if r>0:
                cont.selected = []
                cont.selected.append(contour)
    

if flag == 1:
    root = Tk()
    RobotApp(root, "Bot GUI",Bot)
    root.mainloop()
else:
    root = Tk()
    root.title("Bot GUI")
    text = Label(text="Please connect the Robot")
    text.place(x=20,y=70)
    root.mainloop()