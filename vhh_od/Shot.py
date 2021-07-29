from vhh_od.CustObject import CustObject
from vhh_od.utils import append_dict_as_row
import numpy as np


class Shot(object):
    """
    This class is representing a shot. Each instance of this class is holding the properties of one shot.
    """

    def __init__(self, movie_name, sid, start_pos, end_pos):
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

    def update_obj_classifications(self):
        """
        Updates the class names of objects according to their classification 
        """

        objs = {}

        # Gather all objects according to their tracking id
        for obj in self.object_list:

            # We are only interested in objects that have a person classification
            if obj.person_classification is None:
                continue

            if str(obj.oid) in objs:
                objs[str(obj.oid)].append(obj)
            else:
                objs[str(obj.oid)] = [obj]


        # We do not apply voting as that seems to give worse results
        for obj_list in objs.values():
            for obj in obj_list:
                    obj.update_according_to_person_classification(obj.person_classification)
        return

        # print("BEFORE")
        # for obj_list in objs.values():
        #     for obj in obj_list:
        #         print(obj.person_classification)
        #     print("-----------------")
    
    
        for obj_list in objs.values():
                        # Gather votes
            votes = {}
            for obj in obj_list:
                if obj.person_classification in votes:
                    votes[obj.person_classification] += 1
                else:
                    votes[obj.person_classification] = 1

            # Find winner
            winner_idx = np.argmax(votes.values())
            winner = list(votes.keys())[winner_idx]

            # Update the objects class names and classifications according to the winner
            for obj in obj_list:
                obj.update_according_to_person_classification(winner)
        
        # print("AFTER")
        # for obj_list in objs.values():
        #     for obj in obj_list:
        #         print(obj.person_classification)
        #     print("-----------------")