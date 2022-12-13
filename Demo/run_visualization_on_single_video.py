import vhh_od.visualize as visualize
from vhh_od.Configuration import Configuration
from vhh_od.Video import Video
from vhh_od.utils import *
import sys
import os


"""
Visualizes the results of annotations, creates a video with bounding boxes in the raw results folder specified in config_vhh_od.yaml
Give the name of the video file as parameter, for example '8277.m4v'
Expected to run from the project root folder, for example 'python Demo/run_visualization_on_single_video.py'
Can also give an integer nas a second parameter, then only every n-th frame will be used for the visualization. 
This will decrease the video size to enable faster downloads.
For example 'python Demo/run_visualization_on_single_video.py 8'
"""


if __name__ == "__main__":
    video_file = sys.argv[1]

    config_file = './config/config_vhh_od.yaml'
    config_instance = Configuration(config_file)
    config_instance.loadConfig()
    
    video_file_path = config_instance.path_videos
    csv_results_path = config_instance.path_final_results
    video_results_path = os.path.join(config_instance.path_raw_results, "vis")

    vid_instance = Video()
    vid_instance.load(os.path.join(video_file_path, video_file))
    vid_name = vid_instance.vidName.split(".")[0]

    full_csv_path = os.path.join(csv_results_path, f"{vid_name}.csv")

    if not os.path.isdir(video_results_path):
        os.makedirs(video_results_path)
        visualize.printCustom(f"Created results folder \"{video_results_path}\"", STDOUT_TYPE.INFO)

    # Check if we have a parameter representing which frames to render
    if len(sys.argv) > 2:
        render_every_x_frame = int(sys.argv[2])
    else:
        render_every_x_frame = 1

    visualize.visualize_video(vid_instance, full_csv_path, video_results_path, render_every_x_frame)