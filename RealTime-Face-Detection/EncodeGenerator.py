import cv2
import face_recognition
import pickle
import os

# importing the mode images into a list
folderModePath = 'Images'
modePathList = os.listdir(folderModePath)
imgList = []
for path in modePathList:
    imgList.append(cv2.imread(os.path.join(folderModePath,path)))
print(len(imgList))

