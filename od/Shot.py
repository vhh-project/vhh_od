from od.CustObject import CustObject
from od.utils import append_dict_as_row
import numpy as np


class Shot(object):
    """
    This class is representing a shot. Each instance of this class is holding the properties of one shot.
    """

    def __init__(self, sid, movie_name, start_pos, end_pos):
        """
        Constructor

        :param sid [required]: integer value representing the id of a shot
        :param movie_name [required]: string representing the name of the movie
        :param start_pos [required]: integer value representing the start frame position of the shot
        :param end_pos [required]: integer value representing the end frame position of the shot
        """
        #print("create instance of shot ...");
        self.sid = sid
        self.movie_name = movie_name
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.object_list = []

    def addCustomObject(self, obj: CustObject):
        self.object_list.append(obj)

    def convert2String(self):
        """
        Method to convert class member properties in a semicolon separated string.

        :return: string holding all properties of one shot.
        """
        tmp_str = str(self.sid) + ";" + str(self.movie_name) + ";" + str(self.start_pos) + ";" + str(self.end_pos)
        return tmp_str

    def printShotInfo(self):
        """
        Method to a print summary of shot properties.
        """
        print("------------------------")
        print("shot id: " + str(self.sid))
        print("movie name: " + str(self.movie_name))
        print("start frame: " + str(self.start_pos))
        print("end frame: " + str(self.end_pos))
        print("<<< Object list >>>")
        for obj in self.object_list:
            obj.printObjectInfo()

    def convertObjectList2Dict(self):
        #print("Export objects to csv file ... ")
        dict_l = []
        for idx, obj in enumerate(self.object_list):
            entry_dict = {
                'sid': self.sid,
                'movie_name': self.movie_name,
                'start': self.start_pos,
                'stop': self.end_pos,
                'fid': obj.fid,
                'oid': obj.oid,
                'bb_x1': obj.bb_x1,
                'bb_y1': obj.bb_y1,
                'bb_x2': obj.bb_x2,
                'bb_y2': obj.bb_y2,
                'object_conf': obj.object_conf,
                'class_score': obj.class_score,
                'class_name': obj.object_class_name
            }
            dict_l.append(entry_dict)
        return dict_l
