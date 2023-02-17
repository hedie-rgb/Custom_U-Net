import sys
import os
import cv2
import pathlib
import numpy as np
paper_id = str(sys.argv[1])

#registered resizer

images_path = 'Project/scan-quality/data/handwriting/registereds/registered_'+paper_id
resize_path = 'Project/scan-quality/data/handwriting/resizeds2/resized2_'+paper_id

paths = os.listdir(images_path)
i = 0

for path in paths:
    i+=1
    frame_path = os.path.join(images_path, path)
    if path.find(".DS_Store") != -1: 
        continue
    print("frame", frame_path)
    if not os.path.exists(resize_path):
        os.mkdir(resize_path)
    img = cv2.imread(frame_path)
    img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
    save_path = resize_path+ '/' + str(i)+ "_resized2_area.png"
    cv2.imwrite(save_path, img)
    print("resize",save_path)

#scan resizer

scan_path = 'Project/scan-quality/data/handwriting/scans/scan_'+paper_id+'.png'

img = cv2.imread(scan_path)
img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
save_path = resize_path+ '/scan_resized2_area.png'
cv2.imwrite(save_path, img)