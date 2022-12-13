from vhh_od.OD import OD
import os

config_file = "./config/config_vhh_od.yaml"
od_instance = OD(config_file)

if(od_instance.config_instance.debug_flag == True):
    print("DEBUG MODE activated!")
    stc_results_file = od_instance.config_instance.stc_results_path
    shots_np = od_instance.loadStcResults(stc_results_file)
    print(shots_np)
    max_recall_id = int(shots_np[0][0].split('.')[0])
    od_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)
else:
    results_path = od_instance.config_instance.path_stc_results
    results_file_list = os.listdir(results_path)
    print(results_file_list)

    for file in results_file_list:
        shots_np = od_instance.loadStcResults(results_path + file)
        max_recall_id = int(file.split('.')[0])
        od_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)