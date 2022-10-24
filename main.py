from turtle import distance
import cv2
import math

thres = 0.45

KNOWN_DISTANCE = 60
KNOWN_WIDTH = 8.5

classNames = []
classFile = '/home/y3rsn/Dev/py/stereo/Object-Detector/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = '/home/y3rsn/Dev/py/stereo/Object-Detector/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '/home/y3rsn/Dev/py/stereo/Object-Detector/frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)

net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length)/face_width_in_frame
    return distance


def getObjects(img, thresh, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(
        img, confThreshold=thres, nmsThreshold=nms)
    # print(classIds,bbox)
    distancia = 1
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId-1]
            if className in objects:
                objectInfo.append([className])
                if (draw and classId == 47):
                    #print(box[0]," -  ",box[2]+80," = ",box[2]+80 - box[0] )
                    x1 = box[0]
                    x2 = box[0]+box[2]

                    y1 = box[1]
                    y2 = box[1]

                    try:
                        distancia = math.sqrt((x2-x1)**2+(y2-y1)**2)
                    except:
                        continue

                    #print("distancia: ",distancia)
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(
                        img, "Frasco", (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                    # cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    img = cv2.circle(
                        img, (box[0], box[1]), 10, (0, 0, 255), thickness=2)
                    img = cv2.circle(
                        img, (box[0]+box[2], box[1]), 10, (0, 0, 255), thickness=2)

    return img, objectInfo, distancia


if __name__ == "__main__":
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("/home/y3rsn/Dev/py/stereo/video1.webm")
    cap.set(3, 640)
    cap.set(4, 480)
    # cap.set(10,70)

    success, img = cap.read()
    # ref_img,obj,w = getObjects(img, 0.45, 0.2, objects = [])
    # focal_length_found = FocalLength(KNOWN_DISTANCE, KNOWN_WIDTH,w)#w
    result, objectInfo, w = getObjects(img, 0.45, 0.2, objects=[])
    # print(focal_length_found)
    # W = 6.5# vaso
    W = 8.5  # frasco

    # Finding the Focal Length
    # d = 30# distancia medida
    #f = (w*d)/W
    #print("--> ",f)
    d = 60  # distancia medida
    f = (w*d)/W

    # f = 540 #distancia medida vaso
    print("Focal Length: ", f)
    #cv2.imshow("ref_image", ref_img)

    while True:
        success, img = cap.read()
        #result, objectInfo,distancia = getObjects(img, 0.45, 0.2, objects = [])
        #Distance = Distance_finder(focal_length_found, KNOWN_WIDTH, dist)
        ref_img, obj, w = getObjects(img, 0.45, 0.2, objects=[])
        Distance = (W*f)/w
        #Distance = 30
        cv2.putText(img, f"Distance = {round(Distance,2)} CM",
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # print(objectInfo)
        cv2.imshow("Output", img)

        if cv2.waitKey(50) & 0xFF == ord('q'):

            break

cap.release()
cv2.destroyAllWindows()
