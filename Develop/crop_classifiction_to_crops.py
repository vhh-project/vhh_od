import glob, os
import cv2
import numpy as np
from joblib import Parallel, delayed
import random

crop_information_file = "/data/ext/VHH/datasets/object_classifier/extracted_frames/crop_information.txt"
# crop_information_file = "/data/ext/VHH/datasets/Classifier_data_tracking_2/crop_information.txt"
crops_folder = "/data/ext/VHH/datasets/object_classifier/extracted_frames/"
od_folder = "/data/share/vhh_od_results/vhh_od/final_results"
annotations_file = "/data/ext/VHH/datasets/object_classifier/annotations/annotations_all.csv"
videos_folder = "/data/ext/VHH/release_results/release_v1_3_0/vhh_core/videos_part1"
export_folder = "/data/ext/VHH/datasets/Classifier_data_tracking"

max_size_difference_per_border = 200
max_area_difference_in_percent = 0.2
max_samples_per_annotation = 10

info = []

classes = ["others", "corpse", "soldier", "person_with_kz_uniform"]

random.seed(0)

# Create class folders if necessary
for c in classes:
    path = os.path.join(export_folder, c)
    if not os.path.isdir(path):
        os.mkdir(path)

with open(crop_information_file) as f:
    lines = f.read().split("\n")
    # Get rid of trailing ''
    del lines[-1]
    for line in lines:
        info.append(line.split(","))

# Get annotations
annotations_dict = {}
with open(annotations_file) as f:
    lines = f.read().split("\n")[1:-1]
    for line in lines:
        cols = line.split(";")
        annotations_dict[cols[1]] = cols[3]

# Get files of object detections
od_files = []
for root, dirs, files in os.walk(od_folder):
    for file in files:
        if file.endswith(".csv"):
             od_files.append(os.path.join(root, file))
od_dict = {}
for od_file in od_files:
    od_dict[od_file.split("/")[-1].split(".")[0]] = od_file

# Get files of crops
crops = []
for root, dirs, files in os.walk(crops_folder):
    for file in files:
        if file.endswith(".png"):
             crops.append(os.path.join(root, file))
crops_dict = {}
for crop in crops:
    crops_dict[crop.split("/")[-1]] = crop

# Get files of videos
videos = []
for root, dirs, files in os.walk(videos_folder):
    for file in files:
        if file.endswith(".m4v"):
             videos.append(os.path.join(root, file))
videos_dict = {}
for video in videos:
    videos_dict[video.split("/")[-1].split(".")[0]] = video

def is_roughly_equal(x,y):
    if abs(x-y) <= max_size_difference_per_border:
        return 0
    return 1

def get_crops(od_filename, sid, frame, x1, x2, y1, y2):
    with open(od_filename) as f:
        lines = f.read().split("\n")[1:-2]
        oid = -1

        area = (int(x2)-int(x1))*(int(y2)-int(y1))

        for line in lines:
            cols = line.split(",")
            if cols[1] != sid or cols[4] != frame:
                continue

            x1_new = int(cols[6])
            y1_new = int(cols[7])
            x2_new = int(cols[8])
            y2_new = int(cols[9])

            if is_roughly_equal(x1_new, int(x1)) + is_roughly_equal(x2_new, int(x2)) + \
                            is_roughly_equal(y1_new, int(y1)) + is_roughly_equal(y2_new, int(y2)) == 0:

                print(line)
                oid = cols[5]
                break
        if oid == -1:
            print("Found nothing")
            return {"frames": [int(frame)], "x1": [int(x1)], "x2": [int(x2)], "y1": [int(y1)], "y2": [int(y2)], "oid": oid, "pos":frame}

        frames = []
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        for line in lines:
            cols = line.split(",")
            
            if cols[5] == oid:
                x1_new = int(cols[6])
                y1_new = int(cols[7])
                x2_new = int(cols[8])
                y2_new = int(cols[9])
                if x2_new <= x1_new or y2_new <= y1_new:
                    continue

                area_frame = (x2_new-x1_new)*(y2_new-y1_new)
                # Only check for area if it is not the original frame
                if int(cols[4]) != int(frame):
                    if area / float(area_frame) > 1 + max_area_difference_in_percent or area / float(area_frame) < 1 - max_area_difference_in_percent:
                        continue

                frames.append(int(cols[4]))
                x1.append(x1_new)
                x2.append(x2_new)
                y1.append(y1_new)
                y2.append(y2_new)

        print("Found {0} crops".format(len(frames)))
        return {"frames": frames, "x1": x1, "x2": x2, "y1": y1, "y2": y2, "oid": oid, "pos":frame}

def get_frames(video_path, crop_information, filmname, sid, crop_annotation):
    pos = int(crop_information["pos"])
    frames = crop_information["frames"]
    oid = crop_information["oid"]
    x1 = crop_information["x1"]
    x2 = crop_information["x2"]
    y1 = crop_information["y1"]
    y2 = crop_information["y2"]

    cap= cv2.VideoCapture(video_path)

    if pos not in frames:
        print(pos, frames)

    # If we have too many frames, drop them. Do NOT drop the originally annotated frame
    while len(frames) > max_samples_per_annotation:
        indices = list(range(len(frames)))
        indices.remove(frames.index(pos))
        idx_to_remove = random.choice(indices)
        del frames[idx_to_remove]
        del x1[idx_to_remove]
        del x2[idx_to_remove]
        del y1[idx_to_remove]
        del y2[idx_to_remove]

    if len(frames) == 0:
        return

    for i in frames:
        # Jump to the frame i in the video, the 1 should correspond to CV_CAP_PROP_POS_FRAMES
        cap.set(1, i) 
        _, img  = cap.read()
        idx = frames.index(i)
        img = img[y1[idx]:y2[idx], x1[idx]:x2[idx]]
        path = os.path.join(export_folder, crop_annotation, filmname + "_sid_" + str(sid) + "_pos_" + str(pos) + "_oid_" + str(oid) + "_frame_" + str(i) + ".png")
        cv2.imwrite(path, img)

def crop_magic(crop_info):
    frame_info = crop_info[0].split("_")
    filmname = frame_info[0]
    sid = frame_info[2]
    pos = frame_info[4].split(".")[0]
    crop_filename = crop_info[1].split(".")[0]
    x1 = crop_info[3]
    x2 = crop_info[4]
    y1 = crop_info[5]
    y2 = crop_info[6]

    if filmname in od_dict:
        if filmname in videos_dict:
            if crop_filename not in annotations_dict:
                return
            crop_annotation = annotations_dict[crop_filename]
            crop_information = get_crops(od_dict[filmname], sid, pos, x1, x2, y1, y2)
            if crop_information is not None:
                
                print(crop_annotation)
                print(crop_filename)
                get_frames(videos_dict[filmname], crop_information, filmname, sid, crop_annotation)

import time
start_time = time.time()

Parallel(n_jobs=-1, require='sharedmem')(delayed(crop_magic)(crop_info) for crop_info in info[1:-1])
print("--- %s seconds ---" % (time.time() - start_time))
