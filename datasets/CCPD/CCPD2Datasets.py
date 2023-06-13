import sys

dir_mytest = "/"
sys.path.insert(0, dir_mytest)
from datasets.CCPD.CCPD import CCPD2LPDatasets

if __name__ == '__main__':
    CCPD2LPDatasets(10)