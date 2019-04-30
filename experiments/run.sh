#!/bin/bash

declare -a nets=("inceptionv4" "resnet152" "densenet161" "senet154" "xception" \
                 "dpn" "pnasnet5large" "inceptionresnetv2""mobilenetv2")
declare -A bsize=( ["inceptionv4"]=64 ["resnet152"]=56 ["densenet161"]=40 \
                   ["senet154"]=24 ["xception"]=40 ["dpn"]=24 \
                   ["pnasnet5large"]=8 ["resnext"]=24 ["inceptionresnetv2"]=32 \
                   ["mobilenetv2"]=128 )

TRAIN_ROOT=$ISIC2017_FULL
VAL_ROOT=$ISIC2017_FULL
TEST_ROOT=$ISIC2017_FULL

for i in $(seq 5); do
  TRAIN_CSV=splits/train_$i.csv
  VAL_CSV=splits/val_$i.csv
  TEST_CSV=splits/test_$i.csv
  
  for net in "${nets[@]}"; do
    python3 train.py with \
    train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV val_root=$VAL_ROOT \
    val_csv=$VAL_CSV test_root=$TEST_ROOT test_csv=$TEST_CSV \
    model_name="$net" epochs=500 \
    'aug={"color_contrast": 0.3, "color_saturation": 0.3, \
    "color_brightness": 0.3, "color_hue": 0.1, "rotation": 90, \
    "scale": (0.8, 1.2), "shear": 20, "vflip": True, "hflip": True, \
    "random_crop": True}' batch_size="${bsize[$net]}" \
    early_stopping_patience=16 split_id=$i --name $net 
  done
done
