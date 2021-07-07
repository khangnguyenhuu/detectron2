import os
import cv2
import json
import random
import argparse
import itertools
import numpy as np

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, evaluator
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
import matplotlib.pyplot as plt

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config pipeline file')
    parser.add_argument('--model', type=str, help='Path to model')
    parser.add_argument('--nums_of_class', type=int, help='num of class', default=5)
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--folder_path', type=str, help='image folder')
    parser.add_argument('--output', type=str, help='./output')

    return parser.parse_args()

def predict (args, path_img):
  cfg = get_cfg()
  cfg.merge_from_file(args.config)
  cfg.MODEL.WEIGHTS = args.model

  #cfg.MODEL.WEIGHTS = "mask_rcnn_R_50_FPN_3x_model/model_final.pth"
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8   
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.nums_of_class
  predictor = DefaultPredictor(cfg)
  im = cv2.imread(path_img)
  outputs = predictor(im)
  
  return outputs

if __name__=="__main__":
    args=parse_args()
    json_output = {
      "unknown": 0,
      "xe_2_banh": 0,
      "xe_4_7_cho": 0,
      "xe_tren_7_cho": 0,
      "xe_dac_biet": 0
    }
    class_count=[0, 0, 0, 0, 0]
    for path in os.listdir(args.folder_path):
      path = os.path.join(args.folder_path, path)
      output = predict(args, path)
      boxes = output['instances'].pred_boxes
      scores = output['instances'].scores
      classes = output['instances'].pred_classes
      for i in range (len(classes)):
        if (scores[i] > 0.5):
          class_count[classes[i]]+=1
    json_output['xe_2_banh'] = class_count[1]
    json_output['xe_4_7_cho'] = class_count[2]
    json_output['xe_tren_7_cho'] = class_count[3]
    json_output['xe_dac_biet'] = class_count[4]
    json_output['unknown'] = class_count[0]
    print(json_output)
    with open(os.path.join(args.output, 'output.json') as output:
      json.dump(json_output, output)
