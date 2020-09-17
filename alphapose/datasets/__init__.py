from .coco_det import Mscoco_det
from .concat_dataset import ConcatDataset
from .custom import CustomDataset
from .mscoco import Mscoco
from .mpii import Mpii
from .sjn_210k import Sjn_210k

# 这里的 __all__ 指的是什么？
__all__ = ['CustomDataset', 'Mscoco', 'Mscoco_det', 'Mpii', 'ConcatDataset','Sjn_210k']
