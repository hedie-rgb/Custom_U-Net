import cv2
import numpy as np
import os

def match(frame_path, scan_path):
    frame = cv2.imread(frame_path)
    print(frame.shape)
    G1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scanned = cv2.imread(scan_path)
    print(scanned.shape)
    G2 = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create() 
    keypoints1 = sift.detect(G1,None)
    keypoints2 = sift.detect(G2,None)
    keypoints1, desc1 = sift.compute(G1, keypoints1)
    keypoints2, desc2 = sift.compute(G2, keypoints2)
    bf = cv2.BFMatcher(crossCheck=False)
    matches = bf.knnMatch(desc1,desc2, k=2)
    print(len(matches))
    good_matches = []
    alpha = 0.75
    for m1,m2 in matches:
        if m1.distance < alpha *m2.distance:
            good_matches.append(m1)
    
    print(len(good_matches))
    print(len(good_matches)/len(matches))

    points1 = [keypoints1[m.queryIdx].pt for m in good_matches]
    points1 = np.array(points1,dtype=np.float32)

    points2 = [keypoints2[m.trainIdx].pt for m in good_matches]
    points2 = np.array(points2,dtype=np.float32)
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC,5.0)

    transformed = cv2.warpPerspective(frame, H, (scanned.shape[1], scanned.shape[0]), cv2.BORDER_REPLICATE)


    save_path = frame_path.split(".png")[0] + "_transformed.png"

    
import sys
paper_id = str(sys.argv[1])

images_path = 'Project/scan-quality/data/handwriting/snapshots/snapshot_'+paper_id
scan_path = 'Project/scan-quality/data/handwriting/scans/scan_' + paper_id + '.png'

paths = os.listdir(images_path)

for path in paths:
    frame_path = os.path.join(images_path, path)
    if path.find(".DS_Store") != -1: 
        continue
    print("frame", frame_path)
    print("scan",scan_path)
    match(frame_path, scan_path)


