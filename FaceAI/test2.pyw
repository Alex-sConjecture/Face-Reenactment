#import os
#os.environ['force_plaidML'] = '1'
#
#import sys
#import argparse
#from utils import Path_utils
#from utils import os_utils
#from facelib import LandmarksProcessor
#
#import numpy as np
import cv2
#import time
#import multiprocessing
#import traceback
#from tqdm import tqdm
#from utils.DFLPNG import DFLPNG
#from utils.DFLJPG import DFLJPG
#from utils.cv2_utils import *
#from utils import image_utils
#import shutil

from pathlib import Path

import mathlib

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        ui_path = Path(__file__).parent / "ui" / "main.ui"
        uic.loadUi( str(ui_path) , self)
        
        self.setGeometry(300, 300, 300, 220)
        #self.setWindowTitle('Icon')
        #self.setWindowIcon(QIcon('web.png'))       
        
        a = QPixmap('D:\\DeepFaceLab\\test\\00000.png')
        
        for _ in range(100):
            item = QListWidgetItem()
            icon = QIcon()
            icon.addPixmap(a, QIcon.Normal, QIcon.Off)
            item.setIcon(icon)
            self.listWidget.addItem(item)
        
        #self.workspace_list
        #a = cv2.imread()
        
        #import code
        #code.interact(local=dict(globals(), **locals()))
        
        self.show()

def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())  
    

if __name__ == "__main__":
    main()