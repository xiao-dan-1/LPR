import sys

dir_mytest = "D:\Desktop\license plate recognition\My_LPR"
sys.path.insert(0, dir_mytest)
import os
from utils.CCPD import CCPD2LP

if __name__ == '__main__':
    CCPD2LP(10)