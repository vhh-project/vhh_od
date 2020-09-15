from od.utils import printCustom, STDOUT_TYPE
from od.Configuration import Configuration
from od.Video import Video
from od.Models import *
from od.utils import *
from od.datasets import *
from od.Shot import Shot
from od.CustObject import CustObject

import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
import cv2

import json

class OD(object):
    """
        Main class of shot type classification (stc) package.
    """

    def __init__(self, config_file: str):
        """
        Constructor

        :param config_file: [required] path to configuration file (e.g. PATH_TO/config.yaml)
                                       must be with extension ".yaml"
        """
        print("create instance of stc ... ")

        if (config_file == ""):
            printCustom("No configuration file specified!", STDOUT_TYPE.ERROR)
            exit()

        self.config_instance = Configuration(config_file)
        self.config_instance.loadConfig()

        if (self.config_instance.debug_flag == True):
            print("DEBUG MODE activated!")
            self.debug_results = "/data/share/maxrecall_vhh_mmsi/develop/videos/results/od/develop/"

    def runOnSingleVideo(self, shots_per_vid_np=None, max_recall_id=-1):
        """
        Method to run stc classification on specified video.

        :param shots_per_vid_np: [required] numpy array representing all detected shots in a video
                                 (e.g. sid | movie_name | start | end )
        :param max_recall_id: [required] integer value holding unique video id from VHH MMSI system
        """

        print("run od detector on single video ... ")

        if (type(shots_per_vid_np) == None):
            print("ERROR: you have to set the parameter shots_per_vid_np!")
            exit()

        if (max_recall_id == -1 or max_recall_id == 0):
            print("ERROR: you have to set a valid max_recall_id [1-n]!")
            exit()

        if(self.config_instance.debug_flag == True):
            # load shot list from result file
            shots_np = self.loadSbdResults(self.config_instance.sbd_results_path)
        else:
            shots_np = shots_per_vid_np

        if (len(shots_np) == 0):
            print("ERROR: there must be at least one shot in the list!")
            exit()

        # load video instance
        vid_name = shots_np[0][1]
        vid_instance = Video()
        vid_instance.load(os.path.join(self.config_instance.path_videos, vid_name))

        # prepare numpy shot list
        shot_instance = None
        for s in range(0, len(shots_per_vid_np)):
            # print(shots_per_vid_np[s])
            shot_instance = Shot(sid=int(s + 1),
                                 movie_name=shots_per_vid_np[s][1],
                                 start_pos=int(shots_per_vid_np[s][2]),
                                 end_pos=int(shots_per_vid_np[s][3]))

            vid_instance.addShotObject(shot_obj=shot_instance)

        if (self.config_instance.debug_flag == True):
            num_shots = 3
            offset = 22
        else:
            num_shots = len(vid_instance.shot_list)
            offset = 0

        # prepare transformation for cnn model
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(vid_instance.height), vid_instance.width)),
            transforms.CenterCrop((int(vid_instance.height), int(vid_instance.height))),
            transforms.Resize(self.config_instance.resize_dim),
            #ToGrayScale(),
            transforms.ToTensor(),
            #transforms.Normalize((self.config_instance.mean_values[0] / 255.0,
            #                      self.config_instance.mean_values[1] / 255.0,
            #                      self.config_instance.mean_values[2] / 255.0),
            #                     (self.config_instance.std_dev[0] / 255.0,
            #                      self.config_instance.std_dev[1] / 255.0,
            #                      self.config_instance.std_dev[2] / 255.0))
        ])

        all_tensors_l = vid_instance.getAllFrames(preprocess_pytorch=preprocess)

        # prepare object detection model
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Darknet(config_path=self.config_instance.model_config_path,
                        img_size=self.config_instance.resize_dim).to(device)

        if self.config_instance.path_pre_trained_model.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(self.config_instance.path_pre_trained_model)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(self.config_instance.path_pre_trained_model))

        obj_id = 0
        results_od_l = []
        for idx in range(offset + 0, offset + num_shots):
            shot_id = int(vid_instance.shot_list[idx].sid)
            vid_name = str(vid_instance.shot_list[idx].movie_name)
            start = int(vid_instance.shot_list[idx].start_pos)
            stop = int(vid_instance.shot_list[idx].end_pos)

            if(self.config_instance.debug_flag == True):
                print("-----")
                print(shot_id)
                print(vid_name)
                print(start)
                print(stop)
                print(stop - start)

            shot_tensors = all_tensors_l[start:stop + 1, :, :, :]

            # run od detector
            # prepare pytorch dataloader
            dataset = data.TensorDataset(shot_tensors)  # create your datset
            inference_dataloader = data.DataLoader(dataset=dataset,
                                                   batch_size=self.config_instance.batch_size)

            predictions_l = []
            for i, inputs in enumerate(inference_dataloader):
                input_batch = inputs[0]
                input_batch = Variable(input_batch.type(Tensor))

                # move the input and model to GPU for speed if available
                if torch.cuda.is_available():
                    input_batch = input_batch.to('cuda')
                    model.to('cuda')

                model.eval()
                with torch.no_grad():
                    nms_thres = 0.4
                    output = model(input_batch)
                    detections = non_max_suppression(prediction=output,
                                                     conf_thres=self.config_instance.confidence_threshold,
                                                     nms_thres=nms_thres)
                    predictions_l.extend(detections)

            # prepare results
            for a in range(0, len(predictions_l)):
                frame_id = vid_instance.shot_list[idx].start_pos + a
                frame_based_predictions = predictions_l[a]

                if(self.config_instance.debug_flag == True):
                    print("##################################################################################")

                if (frame_based_predictions is None):
                    results_od_l.append(["None", shot_id, vid_name, start, stop, frame_id,
                                         "None", "None", "None", "None", "None", "None", "None"])

                    if (self.config_instance.debug_flag == True):
                        tmp = str(None) + ";" + str(shot_id) + ";" + str(vid_name) + ";" + str(start) + ";" + str(
                            stop) + ";" + str(frame_id) + ";" + str(None) + ";" + str(None) + ";" + str(
                            None) + ";" + str(None) + ";" + str(None) + ";" + str(None) + ";" + str(None)
                        print(tmp)
                else:
                    #print(str(shot_id) + ";" + str(vid_name) + ";" + str(start) + ";" + str(stop) + ";" + str(frame_id))
                    for b in range(0, len(frame_based_predictions)):
                        obj_id = obj_id + 1
                        pred = frame_based_predictions[b]
                        pred = np.array(pred)
                        results_od_l.append([obj_id, shot_id, vid_name, start, stop, frame_id,
                                             pred[0], pred[1], pred[2], pred[3], pred[4], pred[5], pred[6]])

                        obj_instance = CustObject(oid=b+1,
                                                  fid=frame_id,
                                                  object_class_name="Default",
                                                  conf_score=pred[4],
                                                  bb_x1=pred[0],
                                                  bb_y1=pred[1],
                                                  bb_x2=pred[2],
                                                  bb_y2=pred[3]
                                                  )
                        vid_instance.shot_list[idx].addCustomObject(obj_instance)

                        if (self.config_instance.debug_flag == True):
                            tmp = str(obj_id) + ";" + str(shot_id) + ";" + str(vid_name) + ";" + str(start) + ";" + \
                                  str(stop) + ";" + str(frame_id) + ";" + str(pred[0]) + ";" + str(pred[1]) + ";" + \
                                  str(pred[2]) + ";" + str(pred[3]) + ";" + str(pred[4]) + ";" + str(pred[5]) + ";" + \
                                  str(pred[6])
                            print(tmp)
                ''''''
        if (self.config_instance.debug_flag == True):
            vid_instance.printVIDInfo()

        if (self.config_instance.save_final_results == True):
            vid_instance.export2csv(filepath=self.config_instance.path_final_results + vid_name.split('.')[0] + "." +
                                    self.config_instance.path_postfix_final_results)

        if (self.config_instance.save_raw_results == True):
            print("shots as videos including bbs")

            for shot in vid_instance.shot_list:
                vid_instance.visualizeShotsWithBB(path=self.config_instance.path_raw_results,
                                                  sid=int(shot.sid),
                                                  all_frames_tensors=all_tensors_l,
                                                  save_single_plots_flag=False,
                                                  plot_flag=False,
                                                  boundingbox_flag=True,
                                                  save_as_video_flag=True
                                                  )
        ''''''

    def runModel(self, model, tensor_l):
        """
        Method to calculate stc predictions of specified model and given list of tensor images (pytorch).

        :param model: [required] pytorch model instance
        :param tensor_l: [required] list of tensors representing a list of frames.
        :return: predicted class_name for each tensor frame,
                 the number of hits within a shot,
                 frame-based predictions for a whole shot
        """


    def loadSbdResults(self, sbd_results_path):
        """
        Method for loading shot boundary detection results as numpy array

        .. note::
            Only used in debug_mode.

        :param sbd_results_path: [required] path to results file of shot boundary detection module (vhh_sbd)
        :return: numpy array holding list of detected shots.
        """

        # open sbd results
        fp = open(sbd_results_path, 'r')
        lines = fp.readlines()
        lines = lines[1:]

        lines_n = []
        for i in range(0, len(lines)):
            line = lines[i].replace('\n', '')
            line_split = line.split(';')
            lines_n.append([line_split[0], os.path.join(line_split[1]), line_split[2], line_split[3]])
        lines_np = np.array(lines_n)
        #print(lines_np.shape)

        return lines_np
