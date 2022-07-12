import cv2
import numpy as np
import pandas as pd
import glob

camera_res_dict = {
            '240':(320,240),
            '480':(640,480),
            '600':(800,600),
            '768':(1024, 768),
            '960':(1280, 960),
            '1200':(1600, 1200),
            '1536':(2048, 1536),
            '1944':(2592, 1944),
            '2448':(3264, 2448)}

class Contours():
    def __init__(self) -> None:
        self.best_circ = None
        self.min_len = None
        self.locked = False
        self.selected = []
        self.singular = None
        self.clusters = None

    def mousecallback(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            for contour in self.singular:
                r=cv2.pointPolygonTest(contour, (x,y), False)
                if r>0:
                    self.selected.append(contour)

    def contour_centers(self, contours: tuple) -> list:
        """Function calculates the centers of the inputed contours.

        Args:
            contours (tuple): A tuple of contours to be filtered, normally outputed 
            by cv2.findContours() function.

        Returns:
            list: outputs list of coordinates of the contour centers.
        """
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
        return centers

    def filter_contours(self, contours: tuple, lower: int = 30, upper: int = 400) -> list:
        """Function filteres the inputed contours to be bigger than size eps.

        Args:
            contours (tuple): A tuple of contours to be filtered, normally outputed 
            by cv2.findContours() function.
            eps (int, optional): Threshold for the size of the contours. Defaults to 30.

        Returns:
            list: Outputs list of contours with size greater than eps.
        """    
        filtered_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > lower and cv2.contourArea(contour) < upper:
                filtered_contours.append(contour)
        return filtered_contours

    def find_anchors(self, frame: np.ndarray):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        adjusted = cv2.convertScaleAbs(hsv, alpha=3)
        mask = cv2.inRange(adjusted, (0,100,255), (35,255,255))
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, np.ones((3,3),np.uint8),iterations = 2)

        contours, hierarchy = cv2.findContours(
                dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = self.filter_contours(contours)
        return contours

    def wait_for_anchors(self, cap: cv2.VideoCapture, num_anhors: int = 4, show: bool = False):
        while True:
            ret, frame = cap.read()
            anchors = self.find_anchors(frame)
            cv2.drawContours(frame, anchors, -1,(0,255, 0),2)
            cv2.namedWindow('Anchors', cv2.WINDOW_NORMAL)
            cv2.imshow('Anchors', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
  
        cv2.destroyAllWindows()
        centers = self.contour_centers(anchors)
        return centers

    def get_circles(self, frame: np.ndarray):
        blur = cv2.GaussianBlur(frame,(3,3),0)
        ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        #closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(thresh,kernel,iterations = 3)

        blur2 = cv2.blur(dilation, (7, 7))
        detected_circles = cv2.HoughCircles(image=blur2,
                                            method=cv2.HOUGH_GRADIENT,
                                            dp=1.2,
                                            minDist=500,
                                            param1=100,
                                            param2=50,
                                            minRadius=300,
                                            maxRadius=600
                                            )

        if detected_circles is not None:
            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))
            pt = detected_circles[0][0]
            a, b, r = pt
            
            if self.best_circ is None or r < self.best_circ[2]:
                self.best_circ = pt

            best_center = np.array(self.best_circ[:2])
            curr_center = np.array([a, b])
            #print(best_center, curr_center, np.sqrt(np.sum((best_center - curr_center)**2)))
            if np.sqrt(np.sum((best_center - curr_center)**2)) > 10:
                self.best_circ = pt

    def find_contours(self, frame: np.ndarray, eps: float = 20, offset = 90) -> tuple:
        """General contour finding pipeline of objects inside a circular contour. In our case
        we are looking for objects in a Petri dish.

        Args:
            frame (np.ndarray): frame taken from camera cap, or just an image.
            eps (float, optional): Contour size threshold. Defaults to 20.

        Returns:
            tuple: tuple of circle parameters and contours found in the Petri dish.
        """

        # Convert image to grayscale:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # Normalize the image:
        # nimg = cv2.normalize(gray, None, alpha=0, beta=1,
        #                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # # Apply a little gaussian blur:
        # blurred = cv2.GaussianBlur(nimg, (5, 5), 0)
        # blurred2 = cv2.blur(gray, (7, 7))
        # # Apply binary thresholding:
        # #(T, thresh) = cv2.threshold(blurred, 0.5, 1, cv2.THRESH_BINARY)
        # ret, thresh = cv2.threshold(blurred2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #ret,thresh = cv2.threshold(gray,125,255,cv2.THRESH_BINARY_INV)
        kernel = np.ones((3,3),np.uint8)
        #closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        if not self.locked:
            self.get_circles(gray)
        #result = mask_frame(thresh, self.best_circ, offset)
        result = mask_frame(opening, self.best_circ, offset)
        # Find all the contours in the resulting image.
        contours, hierarchy = cv2.findContours(
            result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # We want to apply a size threshold to the contours.
        self.singular = self.filter_contours(contours, eps, 150)
        self.clusters = self.filter_contours(contours, 150, 1000)

        return self.best_circ


def camera_res(camera_idx: int = 0) -> dict:
    """Function checks which resolutions work with the current camera.

    Args:
        camera_idx (int, optional): Index of the camera. Defaults to 0.

    Returns:
        dict: Dictionary with resolutions as keys, and bool status as values.
    """
    url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
    table = pd.read_html(url)[0]
    table.columns = table.columns.droplevel()
    cap = cv2.VideoCapture(camera_idx)
    resolutions = {}
    for index, row in table[["W", "H"]].iterrows():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, row["W"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, row["H"])
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        resolutions[str(width)+"x"+str(height)] = "OK"
    return resolutions


def video_test(cap: cv2.VideoCapture) -> None:
    """Function for testing the video stream from the current camera. Useful when
    Camera focal length and positions need to be adjusted.

    Args:
        cap (cv2.VideoCapture): video capture object.
    """
    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def set_res(cap: cv2.VideoCapture, res: tuple) -> cv2.VideoCapture:
    """Function to set the resolution of the video capture object.

    Args:
        cap (cv2.VideoCapture): video capture object.
        res (tuple): resolution tuple, like (1024, 768), etc.

    Returns:
        cv2.VideoCapture: modified video capture object.
    """
    cap.set(3, res[0])
    cap.set(4, res[1])
    return cap


def mask_frame(frame, pt, offset):
    a, b, r = pt
    # Create mask to isolate the information in the petri dish.
    mask = np.zeros_like(frame)
    mask = cv2.circle(mask, (a, b), r-offset, (255, 255, 255), -1)
    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Apply the mask to the image.
    result = cv2.bitwise_and(frame.astype('uint8'), mask.astype('uint8'))
    return result


def compute_tf_mtx(mm2pix_dict: dict) -> np.ndarray:
    """Function computes the transformation matrix between real-world
    coordinates and pixel coordinates in an image.

    Args:
        mm2pix_dict (dict): Dictionary mapping real-world coordinates
        to pixel coordinates. Example for four points:
        {(382.76, -113.37): (499, 412),
        (225.27, 94.68): (240, 103),
        (386.5, 91.55): (492, 98),
        (221.25, -110.62): (248, 419)}

    Returns:
        np.ndarray: array that represents the transformation matrix.
    """
    A = np.zeros((2 * len(mm2pix_dict), 6), dtype=float)
    b = np.zeros((2 * len(mm2pix_dict), 1), dtype=float)
    index = 0
    for XY, xy in mm2pix_dict.items():
        X = XY[0]
        Y = XY[1]
        x = xy[0]
        y = xy[1]
        A[2 * index, 0] = x
        A[2 * index, 1] = y
        A[2 * index, 2] = 1
        A[2 * index + 1, 3] = x
        A[2 * index + 1, 4] = y
        A[2 * index + 1, 5] = 1
        b[2 * index, 0] = X
        b[2 * index + 1, 0] = Y
        index += 1
    x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    tf_mtx = np.zeros((3, 3))
    tf_mtx[0, :] = np.squeeze(x[:3])
    tf_mtx[1, :] = np.squeeze(x[3:])
    tf_mtx[-1, -1] = 1
    return tf_mtx


def cam_calibration(chessboardSize: tuple = (9, 7), frameSize: tuple = (1024, 768),
                    squares_sz_mm: float = 20.5, raw_path: str = '../camera_data/raw/'
                    ):
    ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
    # Input the chessboard parameters,

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0],
                           0:chessboardSize[1]].T.reshape(-1, 2)

    objp = objp * squares_sz_mm

    objpoints = []
    imgpoints = []

    images = sorted(glob.glob(raw_path+'image*.png'))

    for idx, image in enumerate(images):

        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If the findChessboardCorners() doesn't work well for you, try the
        # findChessboardCornersSB() alternative. Sometimes it proves to be more
        # robust.
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

        # Check if the algorithm detected any chessboard in the image.
        # If either one of the images gives false, the pair will not be considered
        # for recognition of the corners.
        print(f'i = {idx}, res = {ret}')

        # If found, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)

            cornersL = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(cornersL)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboardSize, cornersL, ret)
            cv2.imshow('Corners', img)
            cv2.waitKey(2000)
        idx += 1
    cv2.destroyAllWindows()

    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frameSize, None, None)
    height, width, channels = img.shape
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, dist, (width, height), 1, (width, height))
