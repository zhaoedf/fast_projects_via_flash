

'''
    for 2-class seg task, convert:
        3-channel [0,255] mask ---> 3-channel [0,1] mask
    notice: 1-channel is not needed and 3-channel works fine for both here and segmentation_pl project.
    [more] using 1-channel mask will cause "RuntimeError: Non RGB images are not supported."
    [cv2] which means that in cv2 you will use gray2bgr in order to save the actual rgb image(nm)
'''


from PIL import Image
import numpy as np
import cv2

labels_color_map = {
    0: (255,255,255),
    1: (0,0,255) # BGR
}
'''
    *对于RGB的label_mask 不能简单的用cv2.threshold进行二值化*，因为似乎threshold的默认操作是取通道-0去做threshold。
    比如这里的task中，labels_color_map如上所示。那么处理之后，0对应的通道-0是255，而1对应的通道0是0，
    这很明显和我们label_id是相反的，这不是我们想要的结果。
    所以正确的做法是如下面这样，通过labels_color_map，结合for循环和bool来赋值。
    
    需要注意的是，通过unique可以看到，二分类的label_mask中的值不总是[255,255,255]/[0,0,255]的，
    可能会有在两个之间线性插值的值，比如[0,0,254]之类的。
    但是[checked]过，这些值很少，可以忽略不计。
'''

cv2_img = cv2.imread('../demo/0.png')
H,W = cv2_img.shape[:2]
res = np.zeros((H, W, 3), dtype=np.uint8)

for label_id, label_val in labels_color_map.items():
    # print((cv2_img==label_val).shape, (cv2_img==label_val)[0,0], (cv2_img==label_val)[257,428])
    res[(cv2_img==label_val)[...,0]] = [label_id]*3


cv2.imwrite('../data/masks/0.png', res)
cv2_img_reload = cv2.imread('../data/masks/0.png')
print(np.unique(cv2_img_reload), cv2_img_reload.shape, cv2_img_reload[0,0]==[0,0,0])





# def preprocess(pil_img, scale):
#     w, h = pil_img.size # indicating that len(pil_img.size) == 2
#     newW, newH = int(scale * w), int(scale * h)
#     assert newW > 0 and newH > 0, 'Scale is too small'
#     pil_img = pil_img.resize((newW, newH))

#     img_nd = np.array(pil_img)
#     # print('img_nd.shape' ,img_nd.shape)

#     if len(img_nd.shape) == 2:
#         img_nd = np.expand_dims(img_nd, axis=2)

#     # HWC to CHW
#     # img_trans = img_nd.transpose((2, 0, 1))
#     if img_nd.max() > 1:
#         # print('!'*25, img_nd.max())
#         img_trans = img_nd // 255
#         # print('img_trans', img_trans.max(), img_trans.dtype)

#     return img_trans




# pil_img = Image.open('../data/masks/0.png').convert('L')
# print('pil_img.size', pil_img.size)
# cv2_img = preprocess(pil_img, 1.0)
# print(cv2_img.shape, cv2_img.dtype, cv2_img.max(),np.unique(cv2_img))
# cv2.imwrite('../data/0.png', cv2_img)
# cv2_img_reload = cv2.imread('../data/0.png')
# print(np.unique(cv2_img_reload), cv2_img_reload.shape)


# pil_img1 = Image.open('../data/masks/0.png').convert('L')
# pil_img2 = Image.open('../data/masks/0.png')
# cv2_img1 = np.array(pil_img1)
# cv2_img2 = np.array(pil_img2)

# print(pil_img1.size, cv2_img1.shape)
# print(pil_img2.size, cv2_img2.shape)


# _, cv2_img = cv2.threshold(cv2_img, 127,255,cv2.THRESH_BINARY)
# print(cv2_img[0,0], cv2_img[257,428])

# print(np.unique(cv2_img))

# for i in range(3):
#     plane = cv2_img[i]
#     cv2_img[i][plane>=122] = 1
#     cv2_img[i][plane<122] = 0
# directly binary thereshold since the values that are not 0 or 255 are *rare*.
# _, cv2_img = cv2.threshold(cv2_img, 127,255,cv2.THRESH_BINARY)
# print(np.unique(cv2_img))
# print(cv2_img[0,0])
# cv2_img = cv2_img//255
# print(cv2_img[0,0])
# cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)