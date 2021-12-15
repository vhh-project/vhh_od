import os
import argparse
import vhh_od.frame_processing as Detector

"""
Runs object detection on a folder of frames and store crops of each boundary box as images.
For more details on the parameters please call this script with the parameter '-h'
Needs to be run from the root project folder (for example via 'python Demo/run_od_on_folder_of_frames.py')
"""

config_file = "./config/config_vhh_od.yaml"


def main():
    #
    # ARGUMENT PARSING
    #

    parser = argparse.ArgumentParser(description="")

    # Get the folder with the pictures as an argument
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('-p', '--path', dest='path', help =
    "The folder containing the pictures for which we want to run object detection,  for example: '-p /data/share/USERNAME/pictures/'. Must be a valid directory.", required=True)
    required_args.add_argument('-o', '--output-path', dest='outpath', help =
    "The folder in which to store crops,  for example: '-p /data/share/USERNAME/crops/'. Must be a valid directory.", required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        raise ValueError("path (-p) must point to a valid directory. Call this script with the '-h' parameter to get information on how to run it")

    if not os.path.isdir(args.outpath):
        raise ValueError("output path ('-o) must point to a valid directory. Call this script with the '-h' parameter to get information on how to run it")

    folder_path = args.path
    output_folder_path = args.outpath
    Detector.run_on_folder_return_crops(folder_path, output_folder_path)

    

    
if __name__ == "__main__":
    main()