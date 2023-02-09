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

# predictions = predictor(image)

class FasterRCNN(nn.Module):
    """Wrap FasterRCNN and return tensors
    """
    def __init__(self, net):
        super(FasterRCNN, self).__init__()
        self.model = net

    def forward(self, x, height, width):
        if x.dim() != 3:
            assert x.shape[0] == 1 and x.dim() == 4
            x = x.squeeze(0)
        inputs = {"image": x.float(), "height": height, "width": width}
        predictions = self.model([inputs])[0]
        instances = predictions['instances']
        return instances.pred_boxes.tensor.floor().int(), instances.scores, instances.pred_classes.int()

m = FasterRCNN(predictor.model).eval().cuda()

with torch.no_grad():
    boxes, scores, labels = m(image_byte, height, width)

onnxfile = "/repos/output/grit.onnx"
targets = ["bbox", "scores", "labels"]
dynamic_axes = {'image': {2 : 'height', 3: 'width'}}
dynamic_axes.update({t: {0: 'i'} for t in targets})
with torch.no_grad():
    torch.onnx.export(m, (image_byte, height, width), onnxfile,
                    verbose=True,
                    input_names=['image', 'height', 'width'],
                    dynamic_axes=dynamic_axes,
                    output_names=targets,
                    opset_version=11)

def optimize_graph(onnxfile, onnxfile_optimized=None, providers=None):
    if providers is None:
        providers = 'CPUExecutionProvider'
    import onnxruntime as rt

    if not onnxfile_optimized:
        onnxfile_optimized = onnxfile[:-5] + "_optimized.onnx"  # ONNX optimizer is broken, using ORT to optimzie
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = onnxfile_optimized
    _ = rt.InferenceSession(onnxfile, sess_options, providers=[providers])
    return onnxfile_optimized

optimize_graph(onnxfile)

import onnxruntime as rt

onnxfile_optimized =  onnxfile[:-5] + "_optimized.onnx"
# sess = rt.InferenceSession(onnxfile, providers=['CPUExecutionProvider'])
sess = rt.InferenceSession(onnxfile, providers=['CUDAExecutionProvider'])
t0 = time.time()
boxes_ort, scores_ort, labels_ort = sess.run(targets, {
    'image': image_byte.cpu().numpy(),
    'height': torch.as_tensor(height).numpy(),
    'width': torch.as_tensor(width).numpy(),
})
print(time.time() - t0)

from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
visualizer = Visualizer(image)
instances = Instances((height, width))
instances.pred_boxes = torch.as_tensor(boxes_ort)
instances.scores = torch.as_tensor(scores_ort)
vis_output = visualizer.draw_instance_predictions(predictions=instances)
vis_output.save("visualization/000000353174_ort.jpg")

image2 = "demo_images/000000497861.jpg"
image2 = cv2.imread(image2)
if predictor.input_format == "RGB":
    # whether the model expects BGR inputs or RGB
    image2 = image2[:, :, ::-1]
height2, width2 = image2.shape[:2]
image2_byte = predictor.aug.get_transform(image2).apply_image(image2).transpose(2, 0, 1)
image2_byte = torch.as_tensor(image2_byte).unsqueeze(0).cuda()

t0 = time.time()
boxes_ort2, scores_ort2, labels_ort2 = sess.run(targets, {
    'image': image2_byte.cpu().numpy(),
    'height': torch.as_tensor(height2).numpy(),
    'width': torch.as_tensor(width2).numpy(),
})
print(time.time() - t0)
instances = Instances((height2, width2))
instances.pred_boxes = torch.as_tensor(boxes_ort2)
instances.scores = torch.as_tensor(scores_ort2)
vis_output = visualizer.draw_instance_predictions(predictions=instances)
vis_output.save("visualization/000000497861_ort.jpg")
