from re import X
import cv2
import numpy as np
import pyelastix, imageio
import elasticdeform
import os

def register(framepath, scanpath):
    image_1 = framepath
    im1 = cv2.imread(image_1) 
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) /255

    image_2 = scanpath
    im2 = cv2.imread(image_2)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) /255

    params = pyelastix.get_default_params(type="BSPLINE")

    params.MaximumNumberOfIterations = 100 
    params.FinalGridSpacingInVoxels = 20 
    params.NumberOfResolutions = 4 
    params.NumberOfHistogramBins = 32   
    im1_deformed, field = pyelastix.register(im1, im2, params)
    field= np.array(field)
    e = cv2.imread(image_1)

    (b, g, r) = cv2.split(e)

    b = elasticdeform.deform_grid(b, field)
    g = elasticdeform.deform_grid(g, field)
    r = elasticdeform.deform_grid(r, field)
    cv2.imwrite(image_1.split(".png")[0] + "_registered.png", cv2.merge([b, g, r]))


import sys
paper_id = str(sys.argv[1])

images_path = 'Project/scan-quality/data/handwriting/transformed_'+paper_id
scan_path = 'Project/scan-quality/data/handwriting/scan_'+paper_id+'.png'

paths = os.listdir(images_path)

for path in paths:
    frame_path = os.path.join(images_path, path)
    if path.find(".DS_Store") != -1: 
        continue
    print("frame", frame_path)
    print("scan",scan_path)
    register(frame_path, scan_path)