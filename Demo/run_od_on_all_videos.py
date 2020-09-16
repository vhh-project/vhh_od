from od.OD import OD
import numpy as np
import os

config_file = "/home/dhelm/VHH_Develop/pycharm_vhh_od/config/config_vhh_od_debug.yaml"
od_instance = OD(config_file)

# get sbd results of all videos in specified folder
results_path = "/data/share/maxrecall_vhh_mmsi/develop/videos/results/sbd/final_results/"
results_file_list = os.listdir(results_path)

for results_file in results_file_list:
    shots_np = od_instance.loadSbdResults(results_path + results_file)

    max_recall_id = int(shots_np[0][1].split('.')[0])
    od_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)

