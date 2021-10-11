


# https://lightning-flash.readthedocs.io/en/stable/api/generated/flash.image.segmentation.serialization.SegmentationLabels.html?highlight=IMAGE_CLASSIFIER_BACKBONES#flash.image.segmentation.serialization.SegmentationLabels.labels_to_image

from flash.image.segmentation.serialization import SegmentationLabels




from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import torch

# 因为process.py中的三通道都是[0,0,0]/[1,1,1]，不管这里的“,0”是否是只读0-channel，都不会影响预期的结果。
cv2_img = cv2.imread('../data/masks/0.png', 0)
print(cv2_img[0,0])
totensor=transforms.ToTensor()

# print(tensor.max(), tensor.shape, tensor.unique())
# print(np.unique(cv2_img))
tensor1 = totensor(cv2_img*255) # *255 is needed, cos totensor do [0,1] normalization
# print(tensor1.unique())

labels_color_map = {
    0: (255,255,255),
    1: (125,0,0)
}

# def labels_to_image(img_labels, labels_map):
#         """Function that given an image with labels ids and their pixels intrensity mapping, creates a RGB
#         representation for visualisation purposes."""
#         assert len(img_labels.shape) == 2, img_labels.shape
#         H, W = img_labels.shape
#         out = torch.empty(3, H, W, dtype=torch.uint8)
#         for label_id, label_val in labels_map.items():
#             if label_id == 0:
#                 assert img_labels[0,0] == 1
#             mask = (img_labels == label_id)
#             # print(mask.unique())
#             for i in range(3):
#                 print(label_val[i])
#                 out[i].masked_fill_(mask, label_val[i])
#                 # print(out[i])
#         return out
    

colorized_mask = SegmentationLabels.labels_to_image(
    img_labels = tensor1[0].clone(),
    labels_map = labels_color_map
)

colorized_mask = colorized_mask.numpy().astype(np.uint8).transpose(1,2,0)
print(colorized_mask.shape, colorized_mask[0,0]==[255,255,255])
cv2.imwrite('../data/colorized_mask.png', colorized_mask)