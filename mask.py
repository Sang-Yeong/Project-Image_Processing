from collections import OrderedDict
# Facial landmarks with dlib, OpenCV, and PythonPython

# import the necessary packages
# imutils 는 OpenCV가 제공하는 기능 중에 좀 복잡하고 사용성이 떨어지는 부분을 잘 보완해 주는 패키지
# translation, rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges, and much more easier with OpenCV
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
# facial의 인덱스를 dictionary로 지정하기
# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17)),
    ("forehead", (68, 81)),
    ("whole_face", (0, 81))
])

# Visualizing facial landmarks with OpenCV and Python
# facial landmarks를 보기 쉽게 색칠
'''
image : The image that we are going to draw our facial landmark visualizations on.
shape : The NumPy array that contains the 68 facial landmark coordinates that map to various facial parts.
colors : A list of BGR tuples used to color-code each of the facial landmark regions.
alpha : A parameter used to control the opacity of the overlay on the original image.
'''


def make_mask(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    # 위에 까는 이미지, 최종 output image 2개 만들기
    face = image.copy()
    element = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220), (200, 200, 200), (0, 0, 0), (255, 255, 255)]

    # loop over the facial landmark regions individually
    # dictaionary로 지정한 facial의 key불러오기
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):

        # grab the (x, y)-coordinates associated with the
        # face landmark
        # 각 facial의 key에 해당하는 value값 불러오기 (j, k)
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]

        if name == "whole_face":  # 얼굴 전체 윤곽(face)

            # cv2.convexHull: 윤곽선(points, contours)의 경계면을 둘러싸는 다각형을 구하는 알고리즘
            # https://www.crocus.co.kr/1288
            # https://hns17.tistory.com/entry/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-ConvexHull-Grahams-Scan
            hull = cv2.convexHull(pts)

            # Contours: 동일한 색 또는 동일한 강도를 가지고 있는 영역의 경계선을 연결한 선
            # https://opencv-python.readthedocs.io/en/latest/doc/15.imageContours/imageContours.html
            # cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType
            # contours: contours정보. contourIdx – contours list type에서 몇번째 contours line을 그릴 것인지. -1 이면 전체
            # thickness – contours line의 두께. 음수이면 contours line의 내부를 채움.
            face = cv2.drawContours(face, [hull], -1, colors[i], -1)


        elif not(name == 'jaw' or name=='forehead'): # 얼굴 요소(element)
            # cv2.convexHull: 윤곽선(points, contours)의 경계면을 둘러싸는 다각형을 구하는 알고리즘
            # https://www.crocus.co.kr/1288
            # https://hns17.tistory.com/entry/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-ConvexHull-Grahams-Scan
            hull = cv2.convexHull(pts)

            # Contours: 동일한 색 또는 동일한 강도를 가지고 있는 영역의 경계선을 연결한 선
            # https://opencv-python.readthedocs.io/en/latest/doc/15.imageContours/imageContours.html
            # cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType
            # contours: contours정보. contourIdx – contours list type에서 몇번째 contours line을 그릴 것인지. -1 이면 전체
            # thickness – contours line의 두께. 음수이면 contours line의 내부를 채움.
            element = cv2.drawContours(element, [hull], -1, colors[-1], -1)


    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    ret, mask_face = cv2.threshold(face, 1, 255, cv2.THRESH_BINARY)

    element = cv2.cvtColor(element, cv2.COLOR_BGR2GRAY)
    ret, mask_element = cv2.threshold(element, 254, 255, cv2.THRESH_BINARY)

    total_mask = mask_face + mask_element
    apply_ori_mask = total_mask
    apply_new_mask = 255 - total_mask

    # cv2.imshow("Image", mask_element)
    # cv2.waitKey(0)

    return apply_ori_mask, apply_new_mask, mask_face, mask_element


def switch_face_color(image, detector, predictor):
    original_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # visualize all facial landmarks with a transparent overlay
        apply_ori_mask, apply_new_mask, mask_face, mask_element = make_mask(image, shape)
        face_mask = cv2.bitwise_and(image, image, mask=apply_new_mask)
        background = cv2.bitwise_and(image, image, mask=mask_face)

        white_img = np.ones(image.shape, dtype=np.uint8)*255
        white_img[apply_new_mask == 255] = (0,0,0)
        # final_image = cv2.bitwise_and(image, apply_ori_mask)

        black_img = np.zeros(image.shape, dtype=np.uint8)*255
        black_img[apply_new_mask == 255] = (86, 123, 179)

        total_img = face_mask + white_img
        final = cv2.bitwise_and(total_img, total_img)




        # cv2.imwrite('./face_landmark/face_mask_2.jpg', face_mask)
        # cv2.imshow("Image", background)
        # cv2.waitKey(0)

        # (x, y) = image.shape[:2]
        # center_image = (int(x / 2), int(y / 2))
        # seamlessclone = cv2.seamlessClone(face_mask, black_img, mask=black_img, coords=center_image, flags=cv2.NORMAL_CLONE)
        # cv2.imshow("Image", seamlessclone)
        # cv2.waitKey(0)
    return background, face_mask, apply_new_mask, mask_element


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
# Instantiating dlib’s HOG-based face detector and loading the facial landmark predictor.

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/mmclab1/Desktop/fakecheck/#image_processing/models/shape_predictor_81_face_landmarks.dat')

# load the input image, resize it, and convert it to grayscale
# Loading and pre-processing our input image
image = cv2.imread('C:/Users/mmclab1/Desktop/fakecheck/#image_processing/face_landmark/00005-frame_3.jpg')
image = imutils.resize(image, width=500)

# '''동영상으로 mask 뽑기'''
# cap = cv2.VideoCapture("./face_landmark/id11_0001.mp4")
# while True:
#     _, image = cap.read()
#     image = imutils.resize(image, width=500)
#     background, face_mask = switch_face_color(image, detector, predictor)
#     new_image = background + face_mask
#
#     cv2.imshow("clone", new_image)
#
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()



background, face_mask, apply_new_mask, mask_element = switch_face_color(image, detector, predictor)
new_image = background+face_mask
# cv2.imwrite('./face_landmark/face_mask.jpg', apply_new_mask)
# cv2.imshow("Image", mask_element)
# cv2.waitKey(0)
