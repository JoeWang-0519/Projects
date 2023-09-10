import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def imrect(im1):
# Perform Image rectification on an 3D array im.
# Parameters: im1: numpy.ndarray, an array with H*W*C representing image.(H,W is the image size and C is the channel)
# Returns: out: numpy.ndarray, rectified image。
#   out =im1
    img = np.array(im1*255, dtype=np.uint8)
    h, w = img.shape[0], img.shape[1]
    # step 1: change to gray image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # step 2: remove noise, utilizing Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=100, sigmaY=100)
    # step 3: detect edge and generate 'binary value' image
    ## to make algorithm more robust, we propose 2 ways to binarized image
    ## here, we should tune the threshold in ordert to generate stable edges
    img_edge = cv2.Canny(img_blur, 100, 300)
    ## second is, by threshold
    _, img_thr = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # step 4: find all contours
    contours_edge, _ = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_thr, _ = cv2.findContours(img_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # step 5: select the contour with largest area
    area_list_edge = []
    area_list_thr = []
    for k in range(len(contours_edge)):
        area_list_edge.append(cv2.contourArea(contours_edge[k]))
    max_index_edge = np.argmax(np.array(area_list_edge))

    for k in range(len(contours_thr)):
        area_list_thr.append(cv2.contourArea(contours_thr[k]))
    max_index_thr = np.argmax(np.array(area_list_thr))
    # step 6: find 4 corners
    perimeter_edge = cv2.arcLength(contours_edge[max_index_edge], True)
    corners_edge = cv2.approxPolyDP(contours_edge[max_index_edge], 0.02 * perimeter_edge, True)

    perimeter_thr = cv2.arcLength(contours_thr[max_index_thr], True)
    corners_thr = cv2.approxPolyDP(contours_thr[max_index_thr], 0.02 * perimeter_thr, True)
    # step 7: learn perspective transformation P
    target_size = (w, h)
    x1, x2, y1, y2 = round(0.8*h), round(0.2*h), round(0.8*w), round(0.2*w)
    if corners_edge.shape[0] == 4:
        corner_starting = np.float32([corners_edge[0][0], corners_edge[1][0], corners_edge[2][0], corners_edge[3][0]])
    else:
        corner_starting = np.float32([corners_thr[0][0], corners_thr[1][0], corners_thr[2][0], corners_thr[3][0]])
    corner_destination = np.float32([[y1, x2], [y2 , x2], [y2, x1], [y1, x1]])
    P = cv2.getPerspectiveTransform(corner_starting, corner_destination)
    # step 8:
    out = cv2.warpPerspective(img, P, dsize=target_size)/255
    return (out)

if __name__ == "__main__":

    # This is the code for generating the rectified output
    # If you have any question about code, please send email to e0444157@u.nus.edu
    # fell free to modify any part of this code for convenience.
    img_names = ['./data/test1.jpg','./data/test2.jpg']
    for name in img_names:
        image = np.array(cv2.imread(name, -1), dtype=np.float32)/255.
        rectificated = imrect(image)
        cv2.imwrite('./data/Result_'+name[7:],np.uint8(np.clip(np.around(rectificated*255,decimals=0),0,255)))
