import argparse, os
from vhh_od.Configuration import Configuration
import vhh_od.frame_processing as Detector


config_file = "./config/config_vhh_od.yaml"

"""
Evaluates how good a model is at object detection.

"""

###
# Argument Parsing
###

parser = argparse.ArgumentParser(description="Evaluates how good a model is")
parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
parser.add_argument("-w", "--weights", type=str, help="Path to checkpoint file (.pth) to convert to a weights file")
parser.add_argument("-n", "--n_imgs", type=int, default=2000, help="Number of images to select from the directory, set to -1 to use all availabe images")
parser.add_argument("-a", "--n_annotations", type=int, default=100, help="Number of samples to manually annotate")

# Get the folder with the pictures as an argument
required_args = parser.add_argument_group('required arguments')
required_args.add_argument('-p', '--path', dest='path', help =
"The folder containing the pictures for which we want to run object detection,  for example: '-p /data/share/USERNAME/pictures/'. Must be a valid directory.", required=True)

args = parser.parse_args()

if not os.path.isdir(args.path):
        raise ValueError("path must point to a valid directory. Call this script with the '-h' parameter to get information on how to run it")

config_instance = Configuration(config_file)
config_instance.loadConfig()

nr_crops, nr_correct_crops, nr_evaluated_crops = Detector.run_on_folder_evaluate_model(args.path, args.n_imgs, args.n_annotations)
print("\n\n\n----------------------Model Details----------------------")
print("Weights: ", config_instance.path_pre_trained_model)
print("Confidence threshold: ", config_instance.confidence_threshold)
print("Nms threshold: ", config_instance.nms_threshold)
print("\n----------------------Model Evaluation----------------------")
print("Number of corrected crops {0} / {1}".format(nr_correct_crops, nr_evaluated_crops))
print("Estimated correctness {0}%".format(round(float(nr_correct_crops)/nr_evaluated_crops, 2)*100))
print("Number of bounding boxes: {0}".format(nr_crops))
print("Number of used images: {0}".format(len(os.listdir(args.path))))