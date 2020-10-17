import pyautogui 
import cv2
import numpy as np
import glob
import random

screenWidth, screenHeight = pyautogui.size()

capture = cv2.VideoCapture(0)
# Load Yolo
net = cv2.dnn.readNet("./yolo_training_final.weights", "./yolov3-tiny-custom.cfg")

# Name custom object
classes = ["1", '2', '3']


layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

font = cv2.FONT_HERSHEY_PLAIN

while True:
    _, frame = capture.read()
    scroll = 0
    img = cv2.resize(frame, None, fx=1, fy=1)
    height, width, channels = img.shape
    standWidth = int(screenWidth / width)
    standHeight = int(screenHeight / height) 

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 1/255, (608, 608), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h, center_x, center_y])  
                confidences.append(float(confidence))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255,10,10)
    stroke = 2

    if len(boxes) > 0:
        pyautogui.moveTo( int(boxes[0][4]) * standWidth , int(boxes[0][5]) * standHeight  )
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.7)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h, center_x, center_y = boxes[i]
            color = (255,0,0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()