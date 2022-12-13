"""
The crops created by crop_classifiction_to_crops.py cannot be used to train YOLO as we would need images and bounding boxes.
This script translates the crops into the data needed to train YOLO
"""

path_crops = r"/data/ext/VHH/datasets/Classifier_data_final"
od_folder = r"/data/share/vhh_od_results/vhh_od/final_results"
videos_folder = r"/data/ext/VHH/release_results/release_v1_3_0/vhh_core/videos_part1"
export_folder = r"/data/ext/VHH/datasets/yolo_datasets/From_annotated_crops"

only_use_classes = ["person_with_kz_uniform"]

classes_to_id = {"others": 0, "soldier": 1, "person_with_kz_uniform": 2, "corpse": 3}

from cProfile import label
from requests.api import get
from vhh_od.helpers import mkdir_if_necessary
import os, glob
import cv2
import pandas as pd
from tqdm import tqdm

# Create required directory structure
splits = ["train", "val", "test"]
for d in ["images", "labels"]:
    mkdir_if_necessary(os.path.join(export_folder, d))
    for split in splits:
        mkdir_if_necessary(os.path.join(export_folder, d, split))

def get_info(img_fullname, label):
    img_name_split = img_fullname.split('.')[0].split('_')
    vid = int(img_name_split[0])
    pos = int(img_name_split[4])
    oid = int(img_name_split[6])
    frame = int(img_name_split[8])
    return {"vid": vid, "pos": pos, "oid": oid, "frame": frame, "label": label}

def get_img_identifier(info_dict):
    return str(info_dict["vid"]) + "_" + str(info_dict["frame"])

def get_name(vid, frame):
    return "{0}_frame_{1}".format(vid, frame)

def get_info_from_identifier(img_id):
    """
    Returns (vid, frame)
    """
    return img_id.split("_")[0], int(img_id.split("_")[1])

def extract_and_store_img(img_id, dir):
    vid, frame = get_info_from_identifier(img_id)
    video_path = os.path.join(videos_folder, vid + ".m4v")
    img_path = os.path.join(dir, get_name(vid, frame) + ".png")

    cap = cv2.VideoCapture(video_path)

    # Jump to the frame i in the video, the 1 should correspond to CV_CAP_PROP_POS_FRAMES
    cap.set(1, frame) 

    _, img  = cap.read()
    height, width, _ = img.shape
    cv2.imwrite(img_path, img)

    return width, height

def create_and_store_labels(img_id, width, height, objects, dir):
    vid, frame = get_info_from_identifier(img_id)

    csv_path = os.path.join(od_folder, str(vid) + ".csv")
    df = pd.read_csv(csv_path)  

    label_path = os.path.join(dir, get_name(vid, frame) + ".txt")
    with open(label_path, "w") as label_file:
        for obj in objects:
            row = df.loc[(df["fid"] == obj["frame"]) & (df["oid"] == obj["oid"])].iloc[0]

            # Normalize coordinates
            x1 = row["bb_x1"] / width
            x2 = row["bb_x2"] / width
            y1 = row["bb_y1"] / height
            y2 = row["bb_y2"] / height

            # To xywh presentation
            bb_x_center = (x2 - x1) / 2 + x1
            bb_y_center = (y2 - y1) / 2 + y1
            bb_width = x2 - x1
            bb_height = y2 - y1

            label_file.write("{0} {1} {2} {3} {4}\n".format(classes_to_id[obj["label"]], bb_x_center, bb_y_center, bb_width, bb_height))

for split in splits:
    print(split)
    labels = {}
    for c in only_use_classes:
        path = os.path.join(path_crops, split, c)
        images = glob.glob(os.path.join(path, "**.png"))
        for img in images:
            info = get_info(os.path.split(img)[-1], c)

            # pos is the position frame of the image that was annotated and frame is the frame at which the image was extracted from the film
            # We can only be sure that every object in an image is annotated if pos == frame
            if info["pos"] != info["frame"]:
                continue

            # Avoid data with invalid OID
            if info["oid"] == -1:
                continue

            image_identifier = get_img_identifier(info)
            if image_identifier in labels:
                labels[image_identifier].append(info)
            else:
                labels[image_identifier] = [info]

    img_output_path = os.path.join(export_folder, "images", split)      
    label_output_path = os.path.join(export_folder, "labels", split)     

    for img_id, objects in tqdm(labels.items()):
        width, height = extract_and_store_img(img_id, img_output_path)
        create_and_store_labels(img_id, width, height, objects, label_output_path)