from detectron.Config import custom_config
import numpy as np
import os, json, random
import cv2

from detectron.detectron2.config import get_cfg
from detectron.detectron2.data.datasets import register_coco_instances
from detectron.detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron.detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
import torch
from detectron.detectron2 import model_zoo
from detectron.detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron.detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron.detectron2.utils.visualizer import Visualizer

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def cv2_imshow(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(), plt.imshow(im), plt.axis('off')


# register_coco_instances("my_dataset_train", {}, "./Dataset-Post/train/train_coco1.json", "./Dataset-Post/train")

for d in ["train", "val"]:
    register_coco_instances("dataset_{}".format(d), {}, "./Dataset-Post/" + d + "/" + d + ".json", "./Dataset-Post/" + d)
    MetadataCatalog.get("dataset_{}".format(d))


def prediction_type(inpu_image):
    print("yessssssssssssssssssssssssss")

    class_name = ['stable', 'crumble', 'shrink', 'progression', 'diffuse enhancement', 'complete response']

    cfg = custom_config(6)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    predictor = DefaultPredictor(cfg)
    outputs = predictor(inpu_image)
    print(outputs['instances'].scores)
    print(outputs)
    v = Visualizer(inpu_image[:, :, ::-1],
                   # metadata=test_metadata,
                   scale=0.8
                   )

    index = outputs['instances'].pred_classes.tolist()
    print(index)
    accuracy = outputs['instances'].scores.tolist()[0]
    # print(class_name[index[0]], "%{:.0f}".format(accuracy * 100))
    name_class = class_name[index[0]]
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2_imshow(out.get_image()[:, :, ::-1])
    print(name_class)
    return out.get_image()[:, :, ::-1], name_class
    # plt.show()


# if __name__ == '__main__':
#     cfg = custom_config(6)
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#
#     # trainer = DefaultTrainer(cfg)
#     # trainer.resume_or_load(resume=False)
#     # trainer.train()
#     # cfg = get_cfg()

#     #evaluator
#     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
#     predictor = DefaultPredictor(cfg)
#     # evaluator = COCOEvaluator("dataset_val", cfg, False, output_dir="./output/")
#     # val_loader = build_detection_test_loader(cfg, "dataset_val")
#     #
#     # evl = inference_on_dataset(predictor.model, val_loader, evaluator)
#     # # print(evl.items())
#     # print('bbox', evl.get('bbox'))
#     # print('Segm', evl.get('segm'))
#     # print(inference_on_dataset(predictor.model, val_loader, evaluator))
#
#
#     # for imageName in glob.glob('./Dataset-Post/val/*jpg'):
#     imageName = './Dataset-Post/val/32CR.jpg'
#     im = cv2.imread(imageName)
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    # metadata=test_metadata,
#                    scale=0.8
#                    )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2_imshow(out.get_image()[:, :, ::-1])
#     plt.show()





# balloon_metadata = MetadataCatalog.get("train")

# dataset_dicts = get_data_dicts("./Dataset-Post/train")
# for d in random.sample(dataset_dicts, 3):
#     print(d)
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2_imshow(out.get_image()[:, :, ::-1])
#     out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2_imshow(out.get_image()[:, :, ::-1])
#     plt.show()


