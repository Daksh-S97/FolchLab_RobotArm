import cv2
import numpy as np
import pandas as pd

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

def find_contours(frame: np.ndarray, eps: float = 20) -> tuple:
    """General contour finding pipeline of objects inside a circular contour. In our case
    we are looking for objects in a Petri dish.

    Args:
        frame (np.ndarray): frame taken from camera cap, or just an image.
        eps (float, optional): Contour size threshold. Defaults to 20.

    Returns:
        tuple: tuple of circle parameters and contours found in the Petri dish.
    """     

    #Convert image to grayscale:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Normalize the image:
    nimg = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #Apply a little gaussian blur:
    blurred = cv2.GaussianBlur(nimg, (5, 5), 0)
    blurred2 = cv2.blur(gray, (3, 3))
    #Apply binary thresholding:
    (T, thresh) = cv2.threshold(blurred, 0.5, 1, cv2.THRESH_BINARY)
    #Detect circles to find the petri dish edges.
    detected_circles = cv2.HoughCircles(image=blurred2, 
                            method=cv2.HOUGH_GRADIENT, 
                            dp=1.2, 
                            minDist=10,
                            param1=50,
                            param2=50,
                            minRadius=200,
                            maxRadius=300                           
                            )

    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        pt = detected_circles[0][0]
        a, b, r = pt[0], pt[1], pt[2]

    #Create mask to isolate the information in the petri dish.
    mask = np.zeros_like(frame)
    mask = cv2.circle(mask, (a,b), r-25, (255,255,255), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #Apply the mask to the image.
    result = cv2.bitwise_and(thresh.astype('uint8'), mask)
    #Find all the contours in the resulting image.
    contours, hierarchy = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #We want to apply a size threshold to the contours.
    sorted_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > eps:
            sorted_contours.append(contour)

    return (a,b,r,sorted_contours)

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
    tf_mtx = np.zeros((3,3))
    tf_mtx[0,:] = np.squeeze(x[:3])
    tf_mtx[1,:] = np.squeeze(x[3:])
    tf_mtx[-1,-1] = 1
    return tf_mtx