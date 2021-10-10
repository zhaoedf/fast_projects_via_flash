

import torch

import flash
from flash.core.data.utils import download_data
from flash.image import SemanticSegmentation, SemanticSegmentationData

# from dataset import SegDataset
from utils.dice_coeff import Dice_coeff

# --------------------------------------------
# 1. Create the DataModule
# --------------------------------------------
nb_classes = 2 
# for 2-class problem, can not be set to 1.
# and the corresponding model outputs mask will the shape of (bs,2,w,h) instead of (bs,w,h)
# 0: neg ; 1:pos
img_size = (512,512)
# https://lightning-flash.readthedocs.io/en/stable/reference/semantic_segmentation.html#semantic-segmentation
# https://lightning-flash.readthedocs.io/en/stable/api/generated/flash.image.segmentation.data.SemanticSegmentationData.html#flash.image.segmentation.data.SemanticSegmentationData
datamodule = SemanticSegmentationData.from_folders(
    train_folder="data/imgs",
    train_target_folder="data/masks",
    test_folder="data/imgs",
    test_target_folder="data/masks",
    # val_split=0.0, # cos demo project only has one sample. even ratio=0.0 will throw an error.
    image_size=img_size,
    num_classes=nb_classes,
)

# --------------------------------------------
# 2. Build the task
# --------------------------------------------
std_lr = 1e-2
optimizer = torch.optim.SGD
optimizer_params = {
    'momentum': 0.99,
    'weight_decay': 5e-4
}

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR
scheduler_params = {
    'milestones': [10,30,60],
    'gamma': 0.1
}

# multi-class problem, so i use CE instead of BCE.
loss_func = torch.nn.functional.cross_entropy

# https://lightning-flash.readthedocs.io/en/stable/api/generated/flash.image.segmentation.model.SemanticSegmentation.html#flash.image.segmentation.model.SemanticSegmentation
model = SemanticSegmentation(
    backbone="resnet18",
    head="unet",
    num_classes=datamodule.num_classes,
    pretrained=False,
    loss_fn=loss_func,
    optimizer=optimizer,
    optimizer_kwargs=optimizer_params,
    scheduler=lr_scheduler,
    scheduler_kwargs=scheduler_params,
    metrics=Dice_coeff()
)
# print(model.backbone)
# from model.backbone_registry import SemanticSegmentationModelHub
# model = SemanticSegmentationModelHub(backbone='zdf/UNet', learning_rate=1e-3)
# seems that i can via "override the step func in Task class which is in flash.core.model.py"
'''
    [backbone & registry]
    https://lightning-flash.readthedocs.io/en/stable/general/registry.html?highlight=registry
    https://lightning-flash.readthedocs.io/en/stable/template/backbones.html?highlight=backbone
'''

# from flash.image.backbones import OBJ_DETECTION_BACKBONES
# from flash.image.classification.backbones import IMAGE_CLASSIFIER_BACKBONES

# print(IMAGE_CLASSIFIER_BACKBONES.available_keys())
# """ out:
# ['adv_inception_v3', 'cspdarknet53', 'cspdarknet53_iabn', 430+.., 'xception71']
# """

# --------------------------------------------
# 3. Create the trainer and finetune the model
# --------------------------------------------
trainer = flash.Trainer(max_epochs=200) # , gpus='9'
trainer.fit(model, datamodule=datamodule)

# --------------------------------------------
# # 4. test a few images!
# --------------------------------------------
trainer.test(model, datamodule=datamodule)

# --------------------------------------------
# 5. Save the model!
# --------------------------------------------
trainer.save_checkpoint("semantic_segmentation_model.pt")