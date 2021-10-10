


# from flash.image.backbones import OBJ_DETECTION_BACKBONES
# from flash.image.classification.backbones import IMAGE_CLASSIFIER_BACKBONES
from flash.image.segmentation.backbones import SEMANTIC_SEGMENTATION_BACKBONES 
from flash.image.segmentation.heads import SEMANTIC_SEGMENTATION_HEADS 

print(SEMANTIC_SEGMENTATION_BACKBONES.available_keys())
print('$'*100)
print(SEMANTIC_SEGMENTATION_HEADS.available_keys())


""" out:
['adv_inception_v3', 'cspdarknet53', 'cspdarknet53_iabn', 430+.., 'xception71']
"""