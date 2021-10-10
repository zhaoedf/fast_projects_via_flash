
'''
    for 2-class seg task, convert:
        3-channel [0,255] mask ---> 3-channel [0,1] mask
    notice: 1-channel is not needed and 3-channel works fine for both here and segmentation_pl project.
'''


from PIL import Image
import numpy as np
import cv2


def preprocess(pil_img, scale):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = pil_img.resize((newW, newH))

    img_nd = np.array(pil_img)
    print(img_nd.shape)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    # img_trans = img_nd.transpose((2, 0, 1))
    if img_nd.max() > 1:
        print('!'*25)
        img_trans = img_nd / 255

    return img_trans




pil_img = Image.open('data/masks/0.png')
cv2_img = preprocess(pil_img, 1.0)
print(cv2_img.shape)
cv2.imwrite('data/0.png',cv2_img)
