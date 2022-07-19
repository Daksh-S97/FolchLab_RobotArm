import sys

from matplotlib.pyplot import plot
from deps import cv_core
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.StartBTN = QPushButton("Start")
        self.StartBTN.clicked.connect(self.StartFeed)
        self.VBL.addWidget(self.StartBTN)

        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

    def StartFeed(self):
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        cont = cv_core.Contours()
        self.ThreadActive = True
        cap = cv2.VideoCapture(0)
        cap = cv_core.set_res(cap, cv_core.camera_res_dict['1200'])
        while self.ThreadActive:
            ret, frame = cap.read()
            if ret:
                plot_img = cv_core.main_pipe(frame, cont)
                Image = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(960, 720, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
                
    def stop(self):
        self.ThreadActive = False
        self.quit()

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())