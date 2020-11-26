from od.Configuration import Configuration
from od.Video import Video
from od.Models import *
from od.utils import *
from od.Shot import Shot
from od.CustObject import CustObject

from deep_sort.deep_sort import DeepSort

import numpy as np
import os
import cv2
from matplotlib import cm
import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms


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

        if (config_file == ""):
            printCustom("No configuration file specified!", STDOUT_TYPE.ERROR)
            exit()

        self.config_instance = Configuration(config_file)
        self.config_instance.loadConfig()

        if (self.config_instance.debug_flag == True):
            print("DEBUG MODE activated!")
            self.debug_results = "/data/share/maxrecall_vhh_mmsi/develop/videos/results/od/develop/"

        printCustom(f"Initializing Deep Sort Tracker...", STDOUT_TYPE.INFO)
        self.use_tracker=True
        self.tracker = DeepSort(model_path="../deep_sort/deep/checkpoint/ckpt.t7")
        printCustom(f"Deep Sort Tracker initialized successfully!", STDOUT_TYPE.INFO)

        self.num_colors = 10
        self.color_map = cm.get_cmap('gist_rainbow', self.num_colors)


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
            printCustom(f"Loading SBD Results from \"{self.config_instance.sbd_results_path}\"...", STDOUT_TYPE.INFO)
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
            num_shots = len(shots_per_vid_np)
            offset = 0

        # load video instance
        vid_name = shots_np[0][0]
        vid_instance = Video()
        vid_instance.load(os.path.join(self.config_instance.path_videos, vid_name))

        # prepare numpy shot list
        shot_instance = None
        for s in range(offset, offset + num_shots):
            # print(shots_per_vid_np[s])
            shot_instance = Shot(sid=int(s + 1),
                                 movie_name=shots_per_vid_np[s][0],
                                 start_pos=int(shots_per_vid_np[s][2]),
                                 end_pos=int(shots_per_vid_np[s][3]))

            vid_instance.addShotObject(shot_obj=shot_instance)


        # prepare object detection model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        printCustom(f"Initializing Model using \"{self.config_instance.model_config_path}\"...", STDOUT_TYPE.INFO)
        model = Darknet(config_path=self.config_instance.model_config_path,
                        img_size=self.config_instance.resize_dim).to(device)

        printCustom(f"Loading Weights from \"{self.config_instance.path_pre_trained_model}\"...", STDOUT_TYPE.INFO)
        if self.config_instance.path_pre_trained_model.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(self.config_instance.path_pre_trained_model)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(self.config_instance.path_pre_trained_model))

        printCustom(f"Loading Class Names from \"{self.config_instance.model_class_names_path}\"... ", STDOUT_TYPE.INFO)
        classes = load_classes(self.config_instance.model_class_names_path)

        # prepare transformation for od model
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((int(vid_instance.height), vid_instance.width)),
            # transforms.CenterCrop((int(vid_instance.height), int(vid_instance.height))),
            transforms.Resize(self.config_instance.resize_dim),
            # ToGrayScale(),
            transforms.ToTensor(),
            # transforms.Normalize((self.config_instance.mean_values[0] / 255.0,
            #                      self.config_instance.mean_values[1] / 255.0,
            #                      self.config_instance.mean_values[2] / 255.0),
            #                     (self.config_instance.std_dev[0] / 255.0,
            #                      self.config_instance.std_dev[1] / 255.0,
            #                      self.config_instance.std_dev[2] / 255.0))
        ])

        resized_dim_y = self.config_instance.resize_dim[0]
        resized_dim_x = self.config_instance.resize_dim[1]

        # Old solution for retrieving all frames at once
        # frames = vid_instance.getAllFrames(preprocess_pytorch=preprocess)
        # all_tensors_l = frames["Tensors"]
        # images_orig = frames["Images"]

        printCustom(f"Starting Object Detection... ", STDOUT_TYPE.INFO)
        printCustom(f"Executing on device {device}...", STDOUT_TYPE.INFO)
        results_od_l = []

        for shot_frames in vid_instance.getFramesByShots(preprocess_pytorch=preprocess):
            shot_tensors = shot_frames["Tensors"]
            images_orig = shot_frames["Images"]

            obj_id = 0

            current_shot = shot_frames["ShotInfo"]

            shot_id = int(current_shot.sid)
            vid_name = str(current_shot.movie_name)
            start = int(current_shot.start_pos)
            stop = int(current_shot.end_pos)

            if(self.config_instance.debug_flag == True):
                print("-----")
                print(f"Video Name: {vid_name}")
                print(f"Shot ID: {shot_id}")
                print(f"Start: {start} / Stop: {stop}")
                print(f"Duration: {stop - start} Frames")

            # run od detector
            predictions_l = self.runModel(model=model, tensor_l=shot_tensors)

            # reset tracker for every new shot
            self.tracker.reset()

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

                    # calculate factors for rescaling bounding boxes
                    im = cv2.cvtColor(images_orig[a], cv2.COLOR_BGR2RGB)
                    y_factor = im.shape[0] / resized_dim_y
                    x_factor = im.shape[1] / resized_dim_x

                    if self.use_tracker:
                        print(frame_based_predictions.shape)

                        # Convert BBoxes from XYXY (corner points) to XYWH (center + width/height) representation
                        x = (frame_based_predictions[:, 0]).cpu().numpy()*x_factor
                        y = (frame_based_predictions[:, 1]).cpu().numpy()*y_factor
                        w = (frame_based_predictions[:, 2]).cpu().numpy()*x_factor - x
                        h = (frame_based_predictions[:, 3]).cpu().numpy()*y_factor - y
                        x = x+w/2
                        y = y+h/2
                        bbox_xywh = np.array([[x[i],y[i],w[i],h[i]] for i in range(len(frame_based_predictions))])

                        # get class confidences
                        cls_conf = frame_based_predictions[:, 5].cpu().numpy()
                        class_predictions = frame_based_predictions[:, 6].cpu().numpy()

                        # Track Objects using Deep Sort tracker
                        # Tracker expects Input as XYWH but returns Boxes as XYXY
                        outputs = self.tracker.update(bbox_xywh, cls_conf, class_predictions, im)
                        print(f"Outputs:\n{outputs}")

                        #for box in bbox_xywh:
                        #    x1 = int(box[0])
                        #    x2 = int(x1 + box[2])
                        #    y1 = int(box[1])
                        #    y2 = int(y1 + box[3])
                        #    color = (0, 255, 0)
                        #    im = cv2.rectangle(im, (x1, y1), (x2, y2), color, 5)

                        num_colors = 10
                        color_map = cm.get_cmap('gist_rainbow', num_colors)

                        if len(outputs) > 0:
                            for box in outputs:
                                x1 = int(box[0])
                                x2 = int(box[2])
                                y1 = int(box[1])
                                y2 = int(box[3])

                                color_idx = box[4] % self.num_colors
                                color = color_map(color_idx)[0:3]
                                color = tuple([int(color[i] * 255) for i in range(len(color))])

                                class_name = classes[int(box[5])]
                                label = f"{class_name} {box[4]}"
                                font_size = 0.5
                                font_thickness = 1
                                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size , font_thickness)[0]

                                # draw bounding box
                                im = cv2.rectangle(im, (x1, y1), (x2, y2), color, 5)

                                # draw text and background
                                cv2.rectangle(im, (x1, y1), (x1 + text_size[0] + 3, y1 + text_size[1] + 4), color, -1)
                                cv2.putText(im, label, (x1, y1 + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                                            [0, 0, 0], font_thickness)

                            cv2.imshow("im", im)
                            cv2.waitKey()



                    #print(str(shot_id) + ";" + str(vid_name) + ";" + str(start) + ";" + str(stop) + ";" + str(frame_id))
                    for b in range(0, len(frame_based_predictions)):
                        obj_id = obj_id + 1
                        pred = frame_based_predictions[b]
                        pred = np.array(pred)
                        results_od_l.append([obj_id, shot_id, vid_name, start, stop, frame_id,
                                             pred[0], pred[1], pred[2], pred[3], pred[4], pred[5], pred[6]])

                        # (x1, y1, x2, y2, object_conf, class_score, class_pred)
                        # TODO: Insert Object ID here!
                        obj_instance = CustObject(oid=b+1,
                                                  fid=frame_id,
                                                  object_class_name=classes[int(pred[6])],
                                                  object_conf=pred[4],
                                                  class_score=pred[5],
                                                  bb_x1=pred[0],
                                                  bb_y1=pred[1],
                                                  bb_x2=pred[2],
                                                  bb_y2=pred[3]
                                                  )
                        current_shot.addCustomObject(obj_instance)

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

            results_path = self.config_instance.path_final_results

            if not os.path.isdir(results_path):
                os.makedirs(results_path)
                printCustom(f"Created results folder \"{results_path}\"", STDOUT_TYPE.INFO)

            filepath = f"{results_path}{vid_name.split('.')[0]}.{self.config_instance.path_postfix_final_results}"
            vid_instance.export2csv(filepath=filepath)

        if (self.config_instance.save_raw_results == True):
            print("shots as videos including bbs")

            results_path = self.config_instance.path_raw_results

            if not os.path.isdir(results_path):
                os.makedirs(results_path)
                printCustom(f"Created results folder \"{results_path}\"", STDOUT_TYPE.INFO)

            for shot in vid_instance.shot_list:
                vid_instance.visualizeShotsWithBB(path=results_path,
                                                  sid=int(shot.sid),
                                                  all_frames_tensors=all_tensors_l,
                                                  save_single_plots_flag=True,
                                                  plot_flag=False,
                                                  boundingbox_flag=True,
                                                  save_as_video_flag=True
                                                  )

    def runModel(self, model, tensor_l):
        """
        Method to calculate stc predictions of specified model and given list of tensor images (pytorch).

        :param model: [required] pytorch model instance
        :param tensor_l: [required] list of tensors representing a list of frames.
        :return: predicted class_name for each tensor frame,
                 the number of hits within a shot,
                 frame-based predictions for a whole shot
        """

        # run od detector

        # prepare pytorch dataloader
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        dataset = data.TensorDataset(tensor_l)  # create your datset
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

        return predictions_l

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
