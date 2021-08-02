import glob, os
import cv2
import numpy as np

crop_information_file = "/data/ext/VHH/datasets/object_classifier/extracted_frames/crop_information.txt"
crops_folder = "/data/ext/VHH/datasets/object_classifier/extracted_frames/"
od_folder = "/data/share/fjogl/vhh_od_dev_results/final_results/"
annotations_file = "/data/ext/VHH/datasets/object_classifier/annotations/annotations_all.csv"
videos_folder = "/data/share/fjogl/vhh_od_dev_results/videos"
export_folder = "/data/ext/VHH/datasets/Classifier_data_tracking"

info = []

classes = ["others", "corpse", "soldier", "person_with_kz_uniform"]

max_size_difference_per_border = 30
max_area_difference_in_percent = 0.1

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
    return abs(x-y) <= max_size_difference_per_border

def get_crops(od_filename, sid, frame, x1, x2, y1, y2):
    with open(od_filename) as f:
        lines = f.read().split("\n")[1:-2]
        oid = -1

        area = (int(x2)-int(x1))*(int(y2)-int(y1))
        for line in lines:
            cols = line.split(",")
            if cols[1] != sid or cols[4] != frame:
                continue
            if is_roughly_equal(int(cols[6]), int(x1)) and is_roughly_equal(int(cols[7]), int(x2)) and \
                is_roughly_equal(int(cols[8]), int(y1)) and is_roughly_equal(int(cols[9]), int(y2)):

                oid = cols[5]
                # print(sid, frame, x1, x2, y1, y2)
                # print("FOUND IT!: ", oid, cols)
                break
        if oid == -1:
            return None

        frames = []
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        for line in lines:
            cols = line.split(",")
            
            if cols[5] == oid:
                if int(cols[7]) <= int(cols[6]) or int(cols[9]) <= int(cols[8]):
                    continue

                area_frame = (int(cols[7])-int(cols[6]))*(int(cols[9])-int(cols[8]))
                if area / float(area_frame) > 1 + max_area_difference_in_percent or area / float(area_frame)< 1 - max_area_difference_in_percent:
                    continue

                frames.append(int(cols[4]))
                x1.append(int(cols[6]))
                x2.append(int(cols[7]))
                y1.append(int(cols[8]))
                y2.append(int(cols[9]))

        print("Found {0} crops".format(len(frames)))
        return {"frames": frames, "x1": x1, "x2": x2, "y1": y1, "y2": y2}

def get_frames(video_path, crop_information, filmname, sid, pos, crop_annotation):
    # print(crop_information)
    frames = crop_information["frames"]
    x1 = crop_information["x1"]
    x2 = crop_information["x2"]
    y1 = crop_information["y1"]
    y2 = crop_information["y2"]

    # print(frames)
    cap= cv2.VideoCapture(video_path)
    i=0
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == False:
            break
        if i in frames:
            idx = frames.index(i)
            print(x1[idx], x2[idx], y1[idx], y2[idx])
            img = img[y1[idx]:y2[idx], x1[idx]:x2[idx]]
            path = os.path.join(export_folder, crop_annotation, filmname + "_sid_" + str(sid) + "_pos_" + str(pos) + "_frame_" + str(i) + ".png")
            print("\t->", path)
            cv2.imwrite(path, img)
            # break
        i+=1

for crop_info in info[1:-1]:
    crop_class = crop_info[2]
    if crop_class != "person":
        continue

    frame_info = crop_info[0].split("_")
    filmname = frame_info[0]
    sid = frame_info[2]
    pos = frame_info[4].split(".")[0]
    crop_filename = crop_info[1]
    crop_path = crops_dict[crop_filename]
    x1 = crop_info[3]
    x2 = crop_info[4]
    y1 = crop_info[5]
    y2 = crop_info[6]

    if filmname in od_dict:
        crop_information = get_crops(od_dict[filmname], sid, pos, x1, x2, y1, y2)
        if crop_information is not None:
            crop_annotation = annotations_dict[crop_filename.split(".")[0]]
            print(crop_annotation)
            print(crop_filename)
            if filmname in videos_dict:
                get_frames(videos_dict[filmname], crop_information, filmname, sid, pos, crop_annotation)

# print(videos_dict)