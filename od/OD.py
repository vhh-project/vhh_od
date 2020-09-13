from od.utils import printCustom, STDOUT_TYPE
from od.Configuration import Configuration
from od.Video import Video
from od.Models import *
from od.utils import *
from od.datasets import *

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

        if (self.config_instance.debug_flag == True):
            num_shots = 3
            offset = 22
        else:
            num_shots = len(shots_np)
            offset = 0

        vid_name = shots_np[0][1]
        vid_instance = Video()
        vid_instance.load(os.path.join(self.config_instance.path_videos, vid_name))

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

        # read all frames of video
        cap = cv2.VideoCapture(self.config_instance.path_videos + "/" + vid_name)
        frame_l = []
        cnt = 0
        while (True):
            cnt = cnt + 1
            ret, frame = cap.read()
            # print(cnt)
            # print(ret)
            # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if (ret == True):
                frame = preprocess(frame)
                frame_l.append(frame) #.transpose((2,0,1)
            else:
                break

        all_tensors_l = torch.stack(frame_l)

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
            # print(shots_np[idx])
            shot_id = int(shots_np[idx][0])
            vid_name = str(shots_np[idx][1])
            start = int(shots_np[idx][2])
            stop = int(shots_np[idx][3])

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
                    #print(type(output))
                    #print(output.size())
                    detections = non_max_suppression(prediction=output,
                                                     conf_thres=self.config_instance.confidence_threshold,
                                                     nms_thres=nms_thres)
                    #print(type(detections))
                    #print(np.array(detections))
                    #print(np.array(detections).shape)
                    #exit()
                    predictions_l.extend(detections)
            print(predictions_l)
            print(len(predictions_l))
            print(type(predictions_l))

            #print(np.array(predictions_l).shape)

            # prepare results
            for a in range(0, len(predictions_l)):
                frame_id = start + a
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

                        if (self.config_instance.debug_flag == True):
                            tmp = str(obj_id) + ";" + str(shot_id) + ";" + str(vid_name) + ";" + str(start) + ";" + \
                                  str(stop) + ";" + str(frame_id) + ";" + str(pred[0]) + ";" + str(pred[1]) + ";" + \
                                  str(pred[2]) + ";" + str(pred[3]) + ";" + str(pred[4]) + ";" + str(pred[5]) + ";" + \
                                  str(pred[6])
                            print(tmp)

        results_od_np = np.array(results_od_l)

        print(results_od_np)
        print(results_od_np.shape)

        if (self.config_instance.save_raw_results == True):
            print("shots as videos including bbs")



        # export results
        self.exportOdResults(str(max_recall_id), results_od_np)

    def saveShotsWithBBs(self, results_file):

        # read and prepare results from csv
        fp = open(self.config_instance.path_final_results + results_file, 'r')
        lines = fp.readlines()
        fp.close()

        lines = lines[1:]
        final_results_l = []
        for line in lines:
            line = line.replace("\n", "")
            line_split = line.split(';')

            line_entries_l = []
            for j in range(0, len(line_split)):
                line_entries_l.append(line_split[j])

            final_results_l.append(line_entries_l)
        results_np = np.array(final_results_l)
        print(results_np)
        print(results_np.shape)


        # read all frames of video


        vid_name = "5.m4v"
        vid_instance = Video()
        vid_instance.load(os.path.join(self.config_instance.path_videos, vid_name))

        # prepare transformation for cnn model
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(vid_instance.height), vid_instance.width)),
            transforms.CenterCrop((int(vid_instance.height), int(vid_instance.height))),
            transforms.Resize(self.config_instance.resize_dim),
        ])

        cap = cv2.VideoCapture(self.config_instance.path_videos + "/" + vid_name)
        frame_l = []
        cnt = 0
        while (True):
            cnt = cnt + 1
            ret, frame = cap.read()
            # print(cnt)
            # print(ret)
            # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if (ret == True):
                frame = preprocess(frame)
                frame_np = np.array(frame)
                frame_l.append(frame_np)  # .transpose((2,0,1)
            else:
                break

        #all_frames_l = torch.stack(frame_l)
        all_frames_np = np.array(frame_l)
        print(all_frames_np.shape)

        #exit()
        ''''''

        # filter results per shot
        shot_ids = np.unique(results_np[:, 1:2])
        #print(shot_ids)
        shot_ids = [23]

        for sid in shot_ids:
            #print("###########")
            #print(sid)
            idx = np.where(results_np[:, 1:2].astype('int') == int(sid))[0]
            #print(idx)

            shot_results_np = results_np[idx]
            #print(shot_results_np)

            for s in range(0, len(shot_results_np)):
                obj_id = shot_results_np[s][0]
                if(obj_id == "None"):
                    #print("no object detected")
                    asdf = 0
                else:
                    obj_id = shot_results_np[s][0]
                    shot_id = shot_results_np[s][1]
                    vid_name = shot_results_np[s][2]
                    start = shot_results_np[s][3]
                    stop = shot_results_np[s][4]
                    frame_id = shot_results_np[s][5]
                    bb_x1 = float(shot_results_np[s][6])
                    bb_y1 = float(shot_results_np[s][7])
                    bb_x2 = float(shot_results_np[s][8])
                    bb_y2 = float(shot_results_np[s][9])
                    #print(obj_id)
                    #print(frame_id)
                    frame = all_frames_np[int(frame_id)]

                    box_w = bb_x2 - bb_x1
                    box_h = bb_y2 - bb_y1

                    print("###############################")
                    print(str(bb_x1) + ";" + str(bb_y1) + ";" + str(bb_x2) + ";" + str(bb_y2))
                    print(str(box_w) + ";" + str(box_h))

                    # save results

                    # Create plot
                    plt.figure()
                    fig, ax = plt.subplots(1)
                    ax.imshow(frame)
                    # Create a Rectangle patch
                    cmap = plt.get_cmap("tab20b")
                    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
                    bbox = patches.Rectangle((bb_x1, bb_y1), box_w, box_h, linewidth=2, edgecolor=colors[0], facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    plt.show()
                    plt.close()




        '''
        for d in range(0, len(results_od_np)):
            obj_id = results_od_np[d][0]
            shot_id = results_od_np[d][1]
            vid_name = results_od_np[d][2]
            start = results_od_np[d][3]
            stop = results_od_np[d][4]
            frame_id = results_od_np[d][5]
            pred[0] = results_od_np[d][6]
            pred[1] = results_od_np[d][7]
            pred[2] = results_od_np[d][8]
            pred[3] = results_od_np[d][9]
        '''


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

    def exportOdResults(self, fName, od_results_np: np.ndarray):
        """
        Method to export od results as csv file.

        :param fName: [required] name of result file.
        :param stc_results_np: numpy array holding the bounding box coordinates as well as the corresponding class names for each frame of one shot of a movie.
        """

        print("export results to csv!")

        if (len(od_results_np) == 0):
            print("ERROR: numpy is empty")
            exit()

        # open stc resutls file
        if (self.config_instance.debug_flag == True):
            fp = open(self.debug_results + "/" + fName + ".csv", 'w')
        else:
            fp = open(self.config_instance.path_final_results + "/" + fName + ".csv", 'w')
        header = "obj_id;sid;vid_name;start;stop;fid;x1;y1;x2;y2;conf_score;score;class_id"
        fp.write(header + "\n")

        for i in range(0, len(od_results_np)):
            tmp_line = str(od_results_np[i][0])
            for c in range(1, len(od_results_np[i])):
                tmp_line = tmp_line + ";" + od_results_np[i][c]
            fp.write(tmp_line + "\n")
