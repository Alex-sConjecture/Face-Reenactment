#import os
#os.environ['force_plaidML'] = '1'
#
#import sys
#import argparse
#from utils import Path_utils
#from utils import os_utils
#from facelib import LandmarksProcessor
#from pathlib import Path
#import numpy as np
#import cv2
#import time
#import multiprocessing
#import traceback
#from tqdm import tqdm
#from utils.DFLPNG import DFLPNG
#from utils.DFLJPG import DFLJPG
#from utils.cv2_utils import *
#from utils import image_utils
#import shutil


import mathlib

import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 300, 220)
        self.setWindowTitle('Icon')
        #self.setWindowIcon(QIcon('web.png'))

        self.show()

def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
    import code
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    main()
