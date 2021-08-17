#!/bin/bash

python Develop/train_classifier.py -e "adam" -o AdamW -a vgg16
python Develop/train_classifier.py -e "adam" -o AdamW -a vgg11
python Develop/train_classifier.py -e "adam_lr_e-5" --lr 0.00001 -o AdamW -a wide_resnet50_2
python Develop/train_classifier.py -e "adam_wd" -o AdamW -a wide_resnet50_2 -w 0.02
python Develop/train_classifier.py -e "sgd" --lr 0.1 -o SGD -a wide_resnet50_2 
python Develop/train_classifier.py -e "wd_0.02" -o AdamW -a vgg16 -w 0.02
python Develop/train_classifier.py -e "wd_0.03" -o AdamW -a wide_resnet50_2 -w 0.03
python Develop/train_classifier.py -e "wd_0.04" -o AdamW -a wide_resnet50_2 -w 0.04
python Develop/train_classifier.py -e "wd_0.05" -o AdamW -a wide_resnet50_2 -w 0.05
python Develop/train_classifier.py -e "wd_0.1" -o AdamW -a wide_resnet50_2 -w 0.1
python Develop/train_classifier.py -e "wd_0.03" -o AdamW -a vgg16 -w 0.03
python Develop/train_classifier.py -e "wd_0.04" -o AdamW -a vgg16 -w 0.04
python Develop/train_classifier.py -e "wd_0.05" -o AdamW -a vgg16 -w 0.05
python Develop/train_classifier.py -e "wd_0.1" -o AdamW -a vgg16 -w 0.1