import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(1, os.path.dirname(os.getcwd()) + "/GenNet_utils/")
from GenNet_utils.GenNet import main


if __name__ == '__main__':
    main()
