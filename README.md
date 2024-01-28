# Tamusa-1-TAMUHack
Hover-Buddy

## Import PySerial for python to run on Arduino IDE. 

#set up the serial line 
ser = serial.Serial('COM4', 9600) 
## Connected to COM4 port set to 9600 may need to adjust.
time.sleep(2)


## Realtime Object Detection


In [1]:
import numpy as np
import cv2
from pathlib import Path 
import torch 

## Pretrained Models 

In [2]: model = torch.hub.load('ultralytics/yolov5',pretrained=True)


Fusing layers... 
YOLOV5s summary: 213 layers, 7225885 parameters, 0 gradients
Adding AutoShape  
## May not need auto shape just need to detect object

In [3]: imgs = [https://ultralytics.com/images/zidane.jpg']
Image(url=imgs[0])

Out[3]: 




In [4]: 
results = model(imgs)
results.print()
results.save(".")

In [5]: 
Image(filename='zidane.jpg')

Out[5]: 

In [6]: 
cap = cv2.VideoCapture(0)
while True: 
    ret, image_np = cap.read()
    results = model(image_np)
    df_result = results.pandas().xyxy[0]
    dict_result = df_result.to_dict()
    scores = list(dict_result["confidence"].values())
    labels = list(dict_result["name"].values())

    list_boxes = list ()
    for dict_item in df_result.to_dict('record'):
        list_boxes.append(list(dict_item.values())[:4])
        count = 0 

for xmin, ymin, xmax, ymax in list_boxes: 
    image_np = cv2.rectangle(image_np, pt1(int(xmin), int(ymin)), pt2=(int(xmax),int(ymax)),\
                color=(255,0,0), thickness=2)
    cv2.putText(image_np, f"{labels[count]}: {round(score[count],2)}", (int(xmin),int(ymin)-10), 
    cv2.FONT_HERSHEY_SIMPLEXF, 0.9, (36,255,12), 2)
    count = count + 1

cv2.imshow('Object Detector', image_np);

if cv2.waitKey(1) & 0xFF == ord('q'):
    cap.release()
    cv2.release()
    cv2.destroyAllWindows()
    break
