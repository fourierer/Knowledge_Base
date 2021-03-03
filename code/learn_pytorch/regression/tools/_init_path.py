import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0,path)

this_dir = osp.dirname(__file__)
# print(this_dir)

regression_path = osp.join(this_dir, '..') # 到达regression目录下
add_path(regression_path)
