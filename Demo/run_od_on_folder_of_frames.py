from vhh_od.OD import OD
import os
import argparse
from vhh_od.Configuration import Configuration

config_file = "./config/config_vhh_od.yaml"

"""
Runs object detection on a folder of frames and visualizes the results.
For more details on the parameters please call this script with the parameter '-h'
Needs to be run from the root project folder (for example via 'python Demo/run_od_on_folder_of_frames.py')
"""


def main():

    config_instance = Configuration(config_file)
    config_instance.loadConfig()

    stc_instance = OD(config_file)

#
# ARGUMENT PARSING
#

    parser = argparse.ArgumentParser(description="")

    # Get the folder with the pictures as an argument
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('-p', '--path', dest='path', help =
    "The folder containing the pictures for which we want to run object detection,  for example: '-p /data/share/USERNAME/pictures/'. Must be a valid directory.", required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        raise ValueError("path must point to a valid directory. Call this script with the '-h' parameter to get information on how to run it")

    stc_instance.runOnAllFramesInFolder(args.path)

if __name__ == "__main__":
    main()