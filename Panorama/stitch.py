from panorama import Panorama
import imutils
import cv2

imageA = cv2.imread('secondPic1.jpg')
imageB = cv2.imread('secondPic2.jpg')
img1 = imutils.resize(imageA, width=400)
img2 = imutils.resize(imageB, width=400)

panorama = Panorama()
(result, matched_points) = panorama.image_stitch([img1, img2], match_status=True)

cv2.imshow("Keypoint Matches", matched_points)
cv2.imshow("Result", result)

cv2.imwrite("Matched_points1.jpg",matched_points)
cv2.imwrite("Panorama_image1.jpg",result)

cv2.waitKey(0)
cv2.destroyAllWindows()
