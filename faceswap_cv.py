import numpy as np
import cv2
from get_facial_landmarks import model, load_landmarks_model, detect_image

def applyAffineTransform(src, srcTriangle, dstTriangle, size):
    warp_mat = cv2.getAffineTransform(np.float32(srcTriangle), np.float32(dstTriangle))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0]+rect[2]:
        return False
    elif point[1] > rect[1]+rect[3]:
        return False
    return True

def calculateDelaunayTriangles(rect, points):
    # Create subdiv and insert points into it
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)

    triangle_list = subdiv.getTriangleList()

    delaunay_triangle = []
    pt = []
    for t in triangle_list:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])


        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            counters = []
            # Get landmarks coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        counters.append(k)

            if len(counters)==3:
                delaunay_triangle.append((counters[0],counters[1],counters[2]))

        pt = []

    return delaunay_triangle

def warpTriangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0]-r1[0]), (t1[i][1]-r1[1])))
        t2_rect.append(((t2[i][0]-r2[0]), (t2[i][1]-r2[1])))
        t2_rect_int.append(((t2[i][0]-r2[0]), (t2[i][1]-r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]: r1[0]+r1[2]]

    size = (r2[2], r2[3])

    img2_rect = applyAffineTransform(img1_rect, t1_rect, t2_rect, size)

    img2_rect = img2_rect * mask

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect


def faceswap_cv(face_image, target_frame):
    img1 = cv2.imread(face_image)
    img2 = cv2.imread(target_frame)
    img1_warped = np.copy(img2)
    # cv2.imshow("face source", img1)
    # cv2.waitKey(0)
    # cv2.imshow("target frame", img2)
    # cv2.waitKey(0)

    # get landmarks
    predictor = load_landmarks_model(model)
    ps1 = detect_image(predictor, face_image)
    ps2 = detect_image(predictor, target_frame)

    points1 = []
    points2 = []
    for i in ps1:
        points1.append((i.x, i.y))
    for i in ps2:
        points2.append((i.x, i.y))

    # with open("mao_00031.jpg.txt", "a") as txt1:
    #     for i in points1:
    #         txt1.writelines(str(i[0])+" "+str(i[1])+"\n")
    # #
    # with open("trump_smile.jpg.txt", "a") as txt2:
    #     for i in points2:
    #         txt2.writelines(str(i[0])+" "+str(i[1])+"\n")

    # find convex hull
    hull1 = []
    hull2 = []

    hull_corners = cv2.convexHull(np.array(points2), returnPoints=False)
    for i in range(0, len(hull_corners)):
        hull1.append(points1[int(hull_corners[i])])
        hull2.append(points2[int(hull_corners[i])])

    rect = (0, 0, img2.shape[1], img2.shape[0])

    dt = calculateDelaunayTriangles(rect, hull2)

    if len(dt) == 0:
        quit()

    # Affine transformtion on Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warpTriangle(img1, img1_warped, t1, t2)

    # Calculate mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

    # Seamless Clone
    output = cv2.seamlessClone(np.uint8(img1_warped), img2, mask, center, cv2.NORMAL_CLONE)

    cv2.imshow("Face Swapped", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    face_image = "./mao_00022.jpg"
    # target_frame = "./duang_000664.jpg"
    target_frame = "./trump_smile.jpg"
    import sys
    face_image = sys.argv[1]
    target_frame = sys.argv[2]
    faceswap_cv(face_image, target_frame)









