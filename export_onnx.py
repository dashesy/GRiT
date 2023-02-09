import os
import time
import cv2
import sys
import torch
import torch.nn as nn

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from grit.config import add_grit_config

cfg = get_cfg()
add_centernet_config(cfg)
add_grit_config(cfg)
cfg.merge_from_file("configs/GRiT_B_DenseCap_ObjectDet.yaml")
cfg.merge_from_list(["MODEL.WEIGHTS", "models/grit_b_densecap_objectdet.pth"])
# Set score_threshold for builtin models
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
cfg.MODEL.TEST_TASK = "ObjectDet"
cfg.MODEL.BEAM_SIZE = 1
cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
cfg.USE_ACT_CHECKPOINT = False
cfg.freeze()

predictor = DefaultPredictor(cfg)

image = "demo_images/000000353174.jpg"
image = cv2.imread(image)

if predictor.input_format == "RGB":
    # whether the model expects BGR inputs or RGB
    image = image[:, :, ::-1]
height, width = image.shape[:2]
image_byte = predictor.aug.get_transform(image).apply_image(image).transpose(2, 0, 1)
image_byte = torch.as_tensor(image_byte).unsqueeze(0).cuda()

predictions = predictor(image)
