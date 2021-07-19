import random
import imutils
import time
import argparse
import cv2
import numpy as np
import os
class yolov4tiny():
    def arg(self):
      ap = argparse.ArgumentParser()
      ap.add_argument("-i", "--image", required=True, help="path to the image input")
      ap.add_argument("-y", "--yolo", required=True, help="path to the yolo")
      ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minium probability to filter weak detection")
      ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when appling non-maximum suppression")
      self.args = vars(ap.parse_args())
      return
    def process(self):
      labelsPath = os.path.sep.join([self.args["yolo"], "coco.names"])
      LABELS = open(labelsPath).read().strip().split("\n")
      np.random.seed(42)
      COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
      weightsPath = os.path.sep.join([self.args["yolo"], "yolov4_tiny.weights"])
      configPath = os.path.sep.join([self.args["yolo"], "yolov4-tiny.cfg"])

      print("[INFO] loading yolo from disk... ")
      net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
      ln = net.getLayerNames()
      ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
      image=cv2.imread(self.args["image"])
      image=imutils.resize(image,height=800, width=1000)
      (H,W) = image.shape[:2]
      ln = net.getLayerNames()
      ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

      blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False )
      net.setInput(blob)
      start = time.time()
      layerOutputs = net.forward(ln)
      end = time.time()

      print("[INFO] YOLO loss: {: .6f} seconds".format(end-start))

      boxes = []
      confidences = []
      classIDs = []
      for output in layerOutputs:
          for detection in output:
             scores = detection[5:]
             classID = np.argmax(scores)
             confidence = scores[classID]
             if confidence > self.args["confidence"]:

                 box = detection[0:4]*np.array([W,H,W,H])
                 (centerX, centerY,width, height)= box.astype("int")
                 x= int(centerX-(width/2))
                 y= int(centerY-(height/2))
                 boxes.append([x, y, int(width), int(height)])
                 confidences.append(float(confidence))
                 classIDs.append(classID)
      idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.args["confidence"], self.args["threshold"])
      if len(idxs) >0:
         for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])
            color = [int(c)for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            text="{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2)
            cv2.imshow("image", image)
            cv2.waitKey(0)
            if cv2.waitKey(1) & 0xff==ord('q'):
               break
obj=yolov4tiny()
obj.arg()
obj.process()
