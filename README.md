# vhh_od

repo based on https://github.com/eriklindernoren/PyTorch-YOLOv3

Tracker adapted from https://github.com/ZQPei/deep_sort_pytorch

# Demo #

* Make sure to have a video (e.g vid.m4v) stored under /videos and a corresponding shot boundary detection result in /results/sbd/final_results (e.g. vid.csv)
* make sure that the vhh_od directory is in your Python-Path
* Settings can be adjusted via config/config_vhh_od_debug.yaml
* ```cd Demo```
* ```python run_od_on_single_video.py```

# Visualization #

* Make sure to have a video (e.g vid.m4v) stored under /videos and a corresponding object detection result in /results/od/final_results (e.g. vid.csv)
* Make sure that the vhh_od directory is in your Python-Path
* Settings can be adjusted via config/vis_config.yaml
* ```cd od```
* ```python visualize.py vid.m4v```