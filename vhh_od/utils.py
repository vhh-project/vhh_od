from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from csv import DictWriter, DictReader


class STDOUT_TYPE:
    INFO = 1
    ERROR = 2

def printCustom(msg: str, type: int):
    if(type == 1):
        print("INFO: " + msg)
    elif(type == 2):
        print("ERROR: " + msg)
    else:
        print("FATAL ERROR: stdout type does not exist!")
        exit()

def add_header(file_name, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Write header
        dict_writer.writeheader()

def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)

def load_csv_as_dict(file_name, field_names):
    dict_l = []
    with open(file_name, 'r') as read_obj:
        reader = DictReader(read_obj, fieldnames=field_names)

        for row in reader:
            dict_l.append(dict(row))

    return dict_l

def remove_csv_export(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
    else:
        print("Warning: The file does not exist!")

def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")
    return names
