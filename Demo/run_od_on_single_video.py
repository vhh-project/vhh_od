from vhh_od.OD import OD
import os

'''
config_file = "/home/dhelm/VHH_Develop/pycharm_vhh_od/config/config_vhh_od_debug.yaml"
od_instance = OD(config_file)


results_path = "/data/share/maxrecall_vhh_mmsi/develop/videos/results/sbd/final_results/"
results_file_list = os.listdir(results_path)
shots_np = od_instance.loadSbdResults(results_path + results_file_list[0])

max_recall_id = 99
od_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)
'''

config_file = "./config/config_vhh_od_debug.yaml"
stc_instance = OD(config_file)

if(stc_instance.config_instance.debug_flag == True):
    print("DEBUG MODE activated!")
    sbd_results_file = stc_instance.config_instance.sbd_results_path
    shots_np = stc_instance.loadSbdResults(sbd_results_file)
    print(shots_np)
    max_recall_id = int(shots_np[0][0].split('.')[0])
    stc_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)
else:
    results_path = "/data/share/datasets/vhh_mmsi_test_db_v3/annotations/sbd/"
    results_file_list = os.listdir(results_path)
    print(results_file_list)

    for file in results_file_list:
        shots_np = stc_instance.loadSbdResults(results_path + file)
        max_recall_id = int(file.split('.')[0])
        stc_instance.runOnSingleVideo(shots_per_vid_np=shots_np, max_recall_id=max_recall_id)

