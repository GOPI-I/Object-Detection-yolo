import cv2
import numpy as np
import face_recognition

imgElong = face_recognition.load_image_file("imageBasic/elon.jpg")
imgElong = cv2.cvtColor(imgElong,cv2.COLOR_BGR2RGB)
imgElonTest = face_recognition.load_image_file("imageBasic/billgates.jpg")
imgElonTest = cv2.cvtColor(imgElonTest,cv2.COLOR_BGR2RGB)

imgElongLoc = face_recognition.face_locations(imgElong)[0]
encodeElong = face_recognition.face_encodings(imgElong)[0]
cv2.rectangle(imgElong,(imgElongLoc[3],imgElongLoc[0]),(imgElongLoc[1],imgElongLoc[2]),(255,0,255),2)

imgElongTestLoc = face_recognition.face_locations(imgElonTest)[0]
encodeElongTest = face_recognition.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest,(imgElongTestLoc[3],imgElongTestLoc[0]),(imgElongTestLoc[1],imgElongTestLoc[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElong],encodeElongTest)
faceDis = face_recognition.face_distance([encodeElong],encodeElongTest)
print(results,faceDis)
cv2.putText(imgElonTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,255),2)
cv2.imshow("Elon musk",imgElong)
cv2.imshow("Elon Test",imgElonTest)

cv2.waitKey(0)