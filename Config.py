from detectron.detectron2.config import get_cfg
from detectron.detectron2 import model_zoo
from detectron.detectron2.data import MetadataCatalog
from detectron.detectron2.data.datasets import register_coco_instances


def custom_config(num_classes):
    # for d in ["train", "val"]:
    #     register_coco_instances("dataset_{}".format(d), {}, "./Dataset-Post/" + d + "/" + d + ".json",
    #                             "./Dataset-Post/" + d)
    #     MetadataCatalog.get("dataset_{}".format(d))

    cfg = get_cfg()

    # get configuration from model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # # Model
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    # cfg.MODEL.RESNETS.DEPTH = 101
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
    cfg.DATASETS.TRAIN = ('dataset_train',)
    cfg.DATASETS.TEST = ('dataset_val',)
    # # DATASETS
    cfg.OUTPUT_DIR = "./detectron/logs"
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64

    return cfg