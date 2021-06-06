import cv2
import numpy as np
import dlib
import time
import imutils

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


'''
step1. 두 이미지 가져오기
- source image(img): swap할 얼굴
- destination image(img2): 추출한 얼굴을 넣는 배경이미지
'''
src_path = "./face_landmark/23.jpg"
img = cv2.imread(src_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray)

back_path = "./face_landmark/22.jpg"
img2 = cv2.imread(back_path)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

'''
step2. 두 이미지의 landmark 찾기(dlib library)
'''
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_81_face_landmarks.dat")

# background image와 같은 크기의 0으로 채워져있는 이미지 만들기
height, width, channels = img2.shape
img2_new_face = np.zeros((height, width, channels), np.uint8)

# Face 1
# dlib을 이용해 얼굴의 landmark를 뽑아 landmarks_points list에 저장하기
faces = detector(img_gray)
for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    for n in range(0, 81):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))

    # list로 구성된 landmarks_points을 numpy array로 바꿔줌.
    points = np.array(landmarks_points, np.int32)

    # landmarks_points를 연결하여 다각형 구하기
    # cv2.convexHull: 윤곽선(points, contours)의 경계면을 둘러싸는 다각형을 구하는 알고리즘
    # https://www.crocus.co.kr/1288
    # https://hns17.tistory.com/entry/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-ConvexHull-Grahams-Scan
    convexhull = cv2.convexHull(points)
    # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)

    #  convexhull에 저장된 좌표로 이루어진 볼록다각형 or 일반 다각형을 color로 채움.
    # 마스크 만들기
    cv2.fillConvexPoly(mask, convexhull, 255)

    # swap할 얼굴 이미지에 마스크 씌우기
    face_image_1 = cv2.bitwise_and(img, img, mask=mask)

    '''
    step3. Triangulation source image(Delaunay triangulation)

    - 삼각분할이 필요한 이유?
    : 얼굴을 swap할 두 이미지에서 뽑아낸 얼굴의 크기와 원근은 동일하지 않을 수 있다.
    이때, 그냥 얼굴을 잘라 크기와 원근을 조절해버린다면 원래 얼굴의 비율이 망가질 수 있다.

    => 얼굴을 삼각형으로 분할하여 삼각형을 교체함으로써 비율 유지를 비롯해 다양한 표정(눈을 감거나 입을 벌릴때)에도 
    새 얼굴의 표현과 일치시킬 수 있다.
    '''
    # Delaunay triangulation
    # cv2.boundingRect: 주어진 점을 감싸는 최소 크기 사각형(바운딩 박스)를 반환함.
    rect = cv2.boundingRect(convexhull)
    # cv2.Subdiv2D: 평면을 삼각형으로 세분화함.
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    # Returns a list of all triangles.
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    '''
    step4. Triangulation destination image
    background 이미지의 Triangulation(삼각측량)은 source 이미지의 Triangulation와 "동일한 패턴"을 가져야 함.
    -> source 이미지의 삼각 측량을 수행 한 뒤, 해당 삼각 측량에서 landmark points의 index(triangles)를 가져와
    background 이미지에 동일한 삼각측량을 복제해야 함.

    ==> 랜드마크 index와 삼각형 index 매칭시킴
    '''
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

# Face 2
# 똑같이 읽어오고 landmark index 저장
faces2 = detector(img2_gray)
for face in faces2:
    landmarks = predictor(img2_gray, face)
    landmarks_points2 = []
    for n in range(0, 81):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points2.append((x, y))

    points2 = np.array(landmarks_points2, np.int32)
    # landmark index로 다각형 그리기
    convexhull2 = cv2.convexHull(points2)

lines_space_mask = np.zeros_like(img_gray)  # source 이미지와 동일한 크기의 0으로 채워진 마스크
lines_space_new_face = np.zeros_like(img2)  # background 이미지와 동일한 크기의 0으로 채워진 마스크

# Triangulation of both faces
# triangles 삼각형 인덱스 -> background image를 삼각 측량함.
# indexes_triangles: source 이미지에서 저장한 삼각형과 landmark를 매핑한 index
for triangle_index in indexes_triangles:
    # Triangulation of the first face
    # source 이미지 얼굴에 대한 landmark point
    tr1_pt1 = landmarks_points[triangle_index[0]]
    tr1_pt2 = landmarks_points[triangle_index[1]]
    tr1_pt3 = landmarks_points[triangle_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

    rect1 = cv2.boundingRect(triangle1)
    (x, y, w, h) = rect1
    # 삼각형 떼어오기
    cropped_triangle = img[y: y + h, x: x + w]
    # 삼각형을 감싸는 사각형
    cropped_tr1_mask = np.zeros((h, w), np.uint8)

    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

    # Lines space
    # source 이미지 삼각형으로 나눈 것 mask로 표현
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
    cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
    lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)

    # Triangulation of second face
    # 두번째 destination image도 삼각형으로 나눠주기
    tr2_pt1 = landmarks_points2[triangle_index[0]]
    tr2_pt2 = landmarks_points2[triangle_index[1]]
    tr2_pt3 = landmarks_points2[triangle_index[2]]
    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

    rect2 = cv2.boundingRect(triangle2)
    (x, y, w, h) = rect2
    cropped_tr2_mask = np.zeros((h, w), np.uint8)
    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                        [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

    '''
    step5. Extract and warp triangles(삼각형 변환)
    두 이미지의 삼각 측량 얻은 후, source image의 삼각형 추출
    background 이미지의 삼각형 좌표 가져와서
    -> source 이미지의 삼각형이 background에서 크기와 원근이 동일하도록 변경할 수 있음.
    '''
    # Warp triangles
    points = np.float32(points)
    points2 = np.float32(points2)

    # getAffineTransform: 선의 평행성은 유지가 되면서 이미지를 변환하는 변환행렬
    M = cv2.getAffineTransform(points, points2)
    # warpAffine: 변환행렬 적용
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

    '''
    step6. Link the warped triangles together
    삼각형을 모두 변환했기 때문에 이제는 연결하는 작업 진행.
    '''

    # Reconstructing destination face
    img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
    img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

'''
step7. Replace the face on the destination image
삼각형으로 모두 합친 이미지 mask로 만들기
'''
# Face swapped (putting 1st face into 2nd face)
img2_face_mask = np.zeros_like(img2_gray)

img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
img2_face_mask = cv2.bitwise_not(img2_head_mask)

# # 눈코입만 남기기
# # load mask
# _, _, _, mask_element = switch_face_color(img2_new_face, detector, predictor)
# cv2.imwrite('./face_landmark/mask_element.jpg', mask_element)
# mask_element_c3 = cv2.imread('./face_landmark/mask_element.jpg')
# img2_new_face = cv2.bitwise_and(img2_new_face, mask_element_c3)
#
# # background이미지에 씌울 마스크: 눈코입만 까맣게
# _, _, _, mask_element = switch_face_color(img2, detector, predictor)
# # cv2.imwrite('./face_landmark/mask_element_background.jpg', mask_element)
# # mask_img2_element = cv2.imread('./face_landmark/mask_element.jpg')
# mask_img2_element = cv2.bitwise_not(mask_element)
#
# img2_head_noface = cv2.bitwise_and(img2, img2, mask=mask_img2_element)
# result = cv2.add(img2_head_noface, img2_new_face)


img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
result = cv2.add(img2_head_noface, img2_new_face)

# cv2.imwrite('./face_landmark/result_1.jpg', result)

# cv2.imshow("image", img2_new_face)
# cv2.waitKey(0)


'''
step8. Seamless Cloning
source 이미지가 background 이미지의 색상과 유사하도록 조정함.
cv2.seamlessClone: 두 이미지의 특징을 살려 알아서 합성하는 기능
'''

(x, y, w, h) = cv2.boundingRect(convexhull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

# dst = cv2.seamlessClone(src, dst, mask, coords, flags, output)
# src: 입력 이미지, 일반적으로 전경
# dst: 대상 이미지, 일반적으로 배경
# mask: 마스크, src에서 합성하고자 하는 영역은 255, 나머지는 0
# coords: src가 놓이기 원하는 dst의 좌표 (중앙)
# flags: 합성 방식
# output(optional): 합성 결과

seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

src_img = src_path.split('/')[-1].split('.')[0]
back_img = back_path.split('/')[-1].split('.')[0]

cv2.imwrite(f'./face_landmark/src_{src_img}_back_{back_img}.jpg', seamlessclone)
# cv2.imshow("seamlessclone", seamlessclone)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()
