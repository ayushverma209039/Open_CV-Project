
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# ## Finding a particular object.
# 
# cv2.matchTemplate(where_bw,template,cv2.TM_CCOEFF)

# In[16]:


where = cv2.imread("C:/Users/User/User/Desktop/Where_is.jpg")
cv2.imshow("Where we need to find.", where)
cv2.waitKey()

template = cv2.imread("C:/Users/User/User/Desktop/temp.jpg")
cv2.imshow("What we need", template)
cv2.waitKey()

where_bw = cv2.cvtColor(where,cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(where_bw,template,cv2.TM_CCOEFF)
min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(result)

topleft = max_loc
bottom_right = (topleft[0]+50,topleft[1]+50)
cv2.rectangle(where,topleft,bottom_right,(0,0,255),5)

cv2.imshow("Located", where)
cv2.waitKey()
cv2.destroyAllWindows()


# ## Finding Corners
# 
# 1. Harris corner detection.

# In[22]:


chess = cv2.imread("C:/Users/User/User/Desktop/chess.jpg")
cv2.imshow("Where we need to find.", chess)
cv2.waitKey()

gray = cv2.cvtColor(chess,cv2.COLOR_BGR2GRAY)

harris_corners = cv2.cornerHarris(gray,3,3,0.05)

kernel = np.ones((7,7),np.uint8)
harris_corners = cv2.dilate(harris_corners, kernel, iterations = 1)

chess[harris_corners > 0.025*harris_corners.max()] = [255,127,127]

cv2.imshow('Harris Corners', chess)
cv2.waitKey()
cv2.destroyAllWindows()


# ## Improved Corner Detection using - Good features to track

# In[24]:


chess = cv2.imread("C:/Users/User/User/Desktop/chess.jpg")
cv2.imshow("Where we need to find.", chess)
cv2.waitKey()
gray = cv2.cvtColor(chess,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,100,0.01,15)

for corner in corners:
    x,y = corner[0]
    x = int(x)
    y = int(y)
    cv2.rectangle(chess,(x-2,y-2),(x+2,y+2),(0,255,0),2)

cv2.imshow('Corners Found', chess)
cv2.waitKey()
cv2.destroyAllWindows()


# #### cv2.goodFeaturesToTrack(input image,number of corners,quality,min distance)

# # SIFT : Scale Invarient Feature Transform
# 
# 

# chess = cv2.imread("C:/Users/User/User/Desktop/chess.jpg")
# cv2.imshow("Where we need to find.", chess)
# cv2.waitKey()
# gray = cv2.cvtColor(chess,cv2.COLOR_BGR2GRAY)
#  
# sift = cv2.SIFT_create()
# 
# keypoints = sift.detect(gray, None)
# print("Number of Keypoints located ", len(keypoints))
# 
# chess = cv2.drawKeypoints(chess,keypoints, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 
# cv2.imshow('Feature Metod - SIFT', chess)
# cv2.waitKey()
# cv2.destroyAllWindows()

# ## SURF, FAST, BRIEF, ORB
# 
# ORB inbuilt so implementation shown.

# In[5]:


chess = cv2.imread("C:/Users/User/User/Desktop/chess.jpg")
cv2.imshow("Where we need to find.", chess)
cv2.waitKey()
gray = cv2.cvtColor(chess,cv2.COLOR_BGR2GRAY)
cv2.imshow("Where we need to find.", gray)
cv2.waitKey()
orb = cv2.ORB_create(5000)

keypoints = orb.detect(gray, None)
print("Number of Keypoints located ", len(keypoints))

cv2.drawKeypoints(chess, keypoints,chess, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Metod - ORB', chess)
cv2.waitKey()
cv2.destroyAllWindows()


# # MINI-Project Object detection using ORB
# 

# In[5]:


def ORB_detector(new_image, image_template):
    
    image1 = cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
    image2 = image_template
    
    orb = cv2.ORB_create(5000,1.2)
    
    (kp1,des1) = orb.detectAndCompute(image1,None)
    (kp2,des2) = orb.detectAndCompute(image2,None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    matches = bf.match(des1,des2)
    
    matches = sorted(matches, key = lambda val: val.distance)
    
    return len(matches)

cap = cv2.VideoCapture(0)

image_template = cv2.imread("C:/Users/User/User/Desktop/4.jpg",0)
## Basically the template is for London Bridge. Take your own photos and check it out. 

while True:
    
    ret, frame = cap.read()
    
    height, width = frame.shape[:2]
    
    tlx = int(width/3) ; tly = int((height/2) + (height/4))
    brx = int(2*width/3); bry = int((height/2) - (height/4))
    
    cv2.rectangle(frame,(tlx,tly),(brx,bry),255,3)
    
    cropped = frame[bry:tly,tlx:brx]
    
    frame = cv2.flip(frame,1)
    
    matches = ORB_detector(cropped, image_template)
    
    output_string = 'Matches =' + str(matches)
    cv2.putText(frame,output_string,(50,450),cv2.FONT_HERSHEY_COMPLEX,2,(250,0,150),2)
    
    threshold = 200
    
    if matches > threshold:
        cv2.rectangle(frame,(tlx,tly),(brx,bry),(0,255,0),3)
        cv2.putText(frame,output_string,(50,450),cv2.FONT_HERSHEY_COMPLEX,2,(250,0,150),2)
        
    cv2.imshow("Object detector using ORB", frame)
    
    if cv2.waitKey(1) == 13:
        break
        
cap.release()
cv2.destroyAllWindows()

