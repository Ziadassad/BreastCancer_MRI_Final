from detectron.detectron2.config import get_cfg
from detectron.detectron2 import model_zoo
from detectron.detectron2.data import MetadataCatalog
from detectron.detectron2.data.datasets import register_coco_instances


def custom_config(num_classes):

    cfg = get_cfg()

    # get configuration from model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    # # Model
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.RESNETS.DEPTH = 101
    cfg.MODEL.DEVICE = "cuda"

    # # Solver
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (210000, 250000)
    cfg.SOLVER.gamma = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 1
    # # Test
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    cfg.TEST.EVAL_PERIOD = 500

    # DATASETS
    cfg.DATASETS.TRAIN = ('dataset_train',)
    cfg.DATASETS.TEST = ('dataset_val',)
    # # DATASETS
    cfg.OUTPUT_DIR = "./detectron/logs"
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64

    return cfg