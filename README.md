# Plugin package: Object Detection and Tracking (ODT)

This package includes all methods to detect and track objects within on image sequence.
This repo is based and is adapted on the following repositories:

    https://github.com/eriklindernoren/PyTorch-YOLOv3
    https://github.com/ZQPei/deep_sort_pytorch

## Package Description

PDF format: [vhh_od_pdf](https://github.com/dahe-cvl/vhh_od/blob/master/ApiSphinxDocumentation/build/latex/vhhpluginpackageshottypeclassificationvhh_stc.pdf)
    
HTML format (only usable if repository is available in local storage): [vhh_od_html](https://github.com/dahe-cvl/vhh_od/blob/master/ApiSphinxDocumentation/build/html/index.html)

## Quick Setup

This package includes a setup.py script and a requirements.txt file which are needed to install this package for custom applications.
The following instructions have to be done to used this library in your own application:

**Requirements:**

   * Ubuntu 18.04 LTS
   * CUDA 10.1 + cuDNN
   * python version 3.6.x

We developed and tested this module with pytorch 1.8-1+cu111 and torchvision 0.9.1+cu111.
   
### 0 Environment Setup (optional)

**Create a virtual environment:**

   * create a folder to a specified path (e.g. /xxx/vhh_od_env/)
   * python3 -m venv /xxx/vhh_od_env/

**Activate the environment:**

   * source /xxx/vhh_od_env/bin/activate

### 1A Install using Pip

The VHH Object Detection and Tracking package is available on [PyPI](https://pypi.org/project/vhh-stc/) and can be installed via ```pip```.

* Update pip and setuptools (tested using pip\==20.2.3 and setuptools==50.3.0)
* ```pip install vhh-od```

Alternatively, you can also build the package from source.

### 1B Install by building from Source

**Checkout vhh_stc repository to a specified folder:**

   * git clone https://github.com/dahe-cvl/vhh_od

**Install the stc package and all dependencies:**

   * Update ```pip``` and ```setuptools``` (tested using pip\==20.2.3 and setuptools==50.3.0)
   * Install the ```wheel``` package: ```pip install wheel```
   * change to the root directory of the repository (includes setup.py)
   * ```python setup.py bdist_wheel```
   * The aforementioned command should create a /dist directory containing a wheel. Install the package using ```python -m pip install dist/xxx.whl```
   
> **_NOTE:_**
You can check the success of the installation by using the commend *pip list*. This command should give you a list
with all installed python packages and it should include *vhh-stc*.

### 2 Install PyTorch

Install a Version of PyTorch depending on your setup. Consult the [PyTorch website](https://pytorch.org/get-started/locally/) for detailed instructions.

### 3 Setup environment variables (optional)

   * source /data/dhelm/python_virtenv/vhh_od_env/bin/activate
   * export CUDA_VISIBLE_DEVICES=1
   * export PYTHONPATH=$PYTHONPATH:/XXX/vhh_od/:/XXX/vhh_od/Develop/:/XXX/vhh_od/Demo/

### 4 Run demo script (optional)

    * Make sure to have a video (e.g vid.m4v) in the videos folder (see "PATH_VIDEOS" in the config file) and a corresponding shot type detection result (e.g. vid.csv) in the stc results folder (see "STC_RESULTS_PATH" in the config file). 
    * make sure that the vhh_od directory is in your Python-Path
    * Settings can be adjusted via config/config_vhh_od.yaml
    * ```cd Demo```
    * ```python run_od_on_single_video.py```

### 5 Visualization (optional)

    * Make sure to have a video (e.g vid.m4v) stored in the videos folder (see "PATH_VIDEOS" in the config file) and a corresponding object detection result in the results folder (see "PATH_FINAL_RESULTS" in the config file).
    * Make sure that the vhh_od directory is in your Python-Path
    * Settings can be adjusted via config/vis_config.yaml
    * ```python Demo/run_visualization_on_single_video.py vid.m4v```
    * This will create a video with bounding boxes in the raw results folder specified in config_vhh_od.yaml

## Release Generation
    
    * Create and checkout release branch: (e.g. v1.1.0): ```git checkout -b v1.1.0```
    * Update version number in setup.py
    * Update Sphinx documentation and release version
    * Make sure that ```pip``` and ```setuptools``` are up to date
    * Install ```wheel``` and ```twine```
    * Build Source Archive and Built Distribution using ```python setup.py sdist bdist_wheel```
    * Upload package to PyPI using ```twine upload dist/*```
