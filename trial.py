import numpy as np
import cv2

arr = cv2.imread("/home/atreyee/Gayathri/Seeing-in-the-Dark-Pytorch/test/1415985340551859.png")
print(arr.shape)
arr = np.expand_dims(arr, axis=2)

arr = np.concatenate([arr, arr, arr], axis=2)

print(arr.shape)