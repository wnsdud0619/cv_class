import numpy as np
import cv2 as cv
import sys

def construct_yolo_v3():
    f = open('coco_names.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]

    model = cv.dnn.readNet('yolov3-weights', 'yolov3.cfg')
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i-1] for i in model.getUnconnectedOutLayers()]

    return model, out_layers, class_names

def yolo_detect(img, yolo_model, out_Layers):
    height, width = img.shape[0], img.shape[1]
    test_img = cv.dnn.blobFromImage(img, 1.0/256, (448, 448), (0,0,0), swapRB=True)
    yolo_model.setInput(test_img)
    outputs = yolo_model.forward(out_layers)

    box, conf, id = [], [], []
    for output in outputs:
        for vec in output:
            scores = vec[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(vec[0]*width), int(vec[1]*height)
                w, h = int(vec[2]*width), int(vec[3]*height)
                x, y = int(center_x - w/2), int(center_y - h/2)
                box.append([x, y, x+w, y+h])
                conf.append(float(confidence))
                id.append(class_id)

    ind = cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)
    objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
    return objects

model, out_layers, class_names = construct_yolo_v3()
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

img = cv.imread('soccer.jpg')
if img is None:
    sys.exit('Image not found.')

res = yolo_detect(img, model, out_layers)

for i in range(len(res)):
    x1, y1, x2, y2, confidence, id = res[i]
    text = str(class_names[id]) + ' %.3f' % confidence
    cv.rectangle(img, (x1, y1), (x2, y2), colors[id], 2)
    cv.putText(img, text, (x1, y1+30), cv.FONT_HERSHEY_PLAIN, 1.5, colors[id], 2)

cv.imshow("Object detection by YOLO v3", img)
cv.waitKey()
cv.destroyAllWindows()
