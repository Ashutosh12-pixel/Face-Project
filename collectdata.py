# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:29:33 2021

@author: Ashutosh
"""

import cv2
import urllib
import numpy as np


face_data=r"haarcascade_frontalface_default.xml"

classifier=cv2.CascadeClassifier(face_data)

url="http://25.162.21.129:8080/shot.jpg"

image_from_url=urllib.request.urlopen(url)

frame=np.array(bytearray(image_from_url.read()),np.uint8)

frame=cv2.imdecode(frame,-1)

faces=classifier.detectMultiScale(frame)

