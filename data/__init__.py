# from .voc import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .coco import COCODetection
from .dummy import DummyDetection
from .data_augment import *
from .config import *
