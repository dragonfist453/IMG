import cv2
import numpy as np 

f = open('../test_imgs/1.jpg','rb')
image = f.read()
f.close()

nparr = np.frombuffer(image, np.uint8)
img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

print(img_np.shape)