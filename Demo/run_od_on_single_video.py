from od.OD import OD
import numpy as np
import os
from od.utils import printCustom, STDOUT_TYPE

config_file = "../config/config_vhh_od_debug.yaml"
od_instance = OD(config_file)


results_path = "../results/sbd/final_results/"
results_file_list = os.listdir(results_path)
first_vid_path = results_path + results_file_list[0]

printCustom(f"Loading SBD Results from \"{first_vid_path}\"...", STDOUT_TYPE.INFO)
shots_np = od_instance.loadSbdResults(first_vid_path)

max_recall_id = 99
od_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)

