import cv2
import sys
import time
import numpy as np

# Get user supplied value
cascPath = sys.argv[1]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# grab webcam
cap = cv2.VideoCapture(0)

while True:
  ret, frame = cap.read()
  start = time.time()
  frame_small = cv2.resize(frame, (400, 267))
  gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
  # Detect faces in the image
  faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(30, 30),
      flags = cv2.cv.CV_HAAR_SCALE_IMAGE
  )
  end = time.time()
  print "time taken = %f" % (end - start)
  print "Found {0} faces!".format(len(faces))
  # Draw a rectangle around the faces
  for (x, y, w, h) in faces:
    cv2.rectangle(frame_small, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv2.imshow('webcam',frame_small)

  #press q or cntl-c with focus on video to quit
  if cv2.waitKey(20) & 0xFF == ord('q'):
    break

  # should display video live, but only do facial detection at some inteval, say 1/10s
  # and then do some basic filtering (perhaps) to establish faces then do facial recognition
  # on prominent / longer standing faces 1-3 times / s depending on how long it takes.
  #time.sleep(0.33)

