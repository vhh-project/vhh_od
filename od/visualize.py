import os
import cv2
import csv
from matplotlib import cm
from od.Video import Video
from od.utils import *

def drawBBox(image, bbox, parameters):
    obj_id = int(bbox[0])
    x1 = int(bbox[1])
    x2 = int(bbox[3])
    y1 = int(bbox[2])
    y2 = int(bbox[4])

    color_idx = obj_id % parameters["num_colors"]
    color = parameters["color_map"](color_idx)[0:3]
    color = tuple([int(color[i] * 255) for i in range(len(color))])

    class_name = "Test"#classes[int(box[5])]
    label = f"{class_name} {obj_id}"
    font = parameters["font"]
    font_size = parameters["font_size"]
    font_thickness = parameters["font_thickness"]
    text_size = cv2.getTextSize(label, font, font_size, font_thickness)[0]

    # draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)

    # draw text and background
    cv2.rectangle(image, (x1, y1), (x1 + text_size[0] + 3, y1 + text_size[1] + 4), color, -1)
    cv2.putText(image, label, (x1, y1 + text_size[1]), font, font_size, [0, 0, 0], font_thickness)

def visualize_video(video: Video, annotations, parameters):
    try:
        annotation = next(annotations)
        annotation_available = True
    except StopIteration:
        annotation_available = False

    if annotation_available == False:
        printCustom(f"CSV-File seems to be empty", STDOUT_TYPE.ERROR)
        return
    elif annotation[0] != f"{vid_id}.{vid_format}":
        printCustom(f"CSV-File does not seem to match Video", STDOUT_TYPE.ERROR)
        return

    frame_idx = 0
    video_writer = None

    for frame in video.loadVideoByFrame():
        image = frame["Images"]

        if video_writer is None:
            frame_size = (image.shape[1], image.shape[0])
            print(frame_size)
            video_writer = cv2.VideoWriter(parameters["result_path"], cv2.VideoWriter_fourcc(*"MJPG"), 12, frame_size)
            printCustom(f"Writing Video to {parameters['result_path']}", STDOUT_TYPE.INFO)

        objects = 0
        while annotation_available and int(annotation[4]) == frame_idx:
            annotation = np.array(annotation)
            drawBBox(image, annotation[[5,6,7,8,9,12]], parameters)
            try:
                annotation = next(csv_reader)
            except StopIteration:
                annotation_available = False
            objects += 1

        #if objects > 0:
            #printCustom(f"Found {objects} Objects in Frame {frame_idx}", STDOUT_TYPE.INFO)

        video_writer.write(image)

        frame_idx += 1
    video_writer.release()
    printCustom(f"Visualization done. Wrote {frame_idx} frames to \"{parameters['result_path']}", STDOUT_TYPE.INFO)


# ---- Vis Parameters

num_colors = 10
cm_name = 'gist_rainbow'
color_map = cm.get_cmap(cm_name, num_colors)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.5
font_thickness = 1

# ---- Paths

results_path = "../results/od"
csv_results_path = os.path.join(results_path, "final_results")
video_results_path = os.path.join(results_path, "vis")
video_file_path = "../videos"

vid_id = 35
vid_format = "m4v"

if __name__ == "__main__":

    # TODO: read params from config

    print("Visualizing...")

    vid_instance = Video()
    full_video_path = os.path.join(video_file_path, f"{vid_id}.{vid_format}")
    vid_instance.load(full_video_path)

    full_csv_path = os.path.join(csv_results_path, f"{vid_id}_dstrack.csv")
    csv_file = open(full_csv_path, "r")
    csv_reader = csv.reader(csv_file, delimiter=",")

    if not os.path.isdir(video_results_path):
        os.makedirs(video_results_path)
        printCustom(f"Created results folder \"{video_results_path}\"", STDOUT_TYPE.INFO)

    parameters = {"num_colors": num_colors,
                  "color_map" : color_map,
                  "font" : font,
                  "font_size" : font_size,
                  "font_thickness" : font_thickness,
                  "result_path" : os.path.join(video_results_path, f"{vid_id}_results.avi")}

    visualize_video(vid_instance, csv_reader, parameters)
