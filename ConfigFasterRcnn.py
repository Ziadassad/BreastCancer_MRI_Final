from detectron.detectron2.config import get_cfg
from detectron.detectron2 import model_zoo
import os
from detectron.detectron2.data import MetadataCatalog
from detectron.detectron2.data.datasets import register_coco_instances
from detectron.detectron2.engine import DefaultTrainer
import sys

def custom_config3(num_classes):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.RESNETS.DEPTH = 101
    cfg.MODEL.DEVICE = "cuda"

    # # Solver
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (210000, 250000)
    cfg.SOLVER.gamma = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 2
    # # Test
    cfg.TEST.DETECTIONS_PER_IMAGE = 20

    # DATASETS
    cfg.DATASETS.TRAIN = ('faster_train',)
    cfg.DATASETS.TEST = ('faster_val',)
    # # DATASETS
    cfg.OUTPUT_DIR = "./detectron/FasterRcnn"
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64

    return cfg

for d in ["train", "val"]:
    register_coco_instances("faster_{}".format(d), {}, "./detectron/WLE/" + d + "/" + d + ".json", "./detectron/WLE/" + d)
    MetadataCatalog.get("faster_{}".format(d))

if __name__ == '__main__':
    cfg = custom_config3(7)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    # trainer.train()
