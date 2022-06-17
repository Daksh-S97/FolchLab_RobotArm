import numpy as np
import cv2
from deps import cv_core



# Create videoCapture objects for both video streams.
cap = cv2.VideoCapture(0)
cap = cv_core.set_res(cap, (1600, 1200))
num = 0

# Set infinite loop to capture images from video.
while True:
    # We use the .grab() method to reduce the lag between the two videos.
    ret, frame = cap.read()

    k = cv2.waitKey(5)
    if k == 27:
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        # Put whatever directory is convenient for you.
        cv2.imwrite('camera_data/raw/image' +
                    str(num) + '.png', frame)
        # cv2.imwrite('image'+
        #     str(num) + '.png', frame)
        print("images saved!")
        num += 1
    # Displaying the capture in a single window, more convenient.
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', frame)

cv2.destroyAllWindows()