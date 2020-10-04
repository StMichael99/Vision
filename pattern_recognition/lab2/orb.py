import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

FLANN_INDEX_LSH= 6
index_params= dict(algorithm = FLANN_INDEX_LSH,table_number = 6,key_size = 12,multi_probe_level = 1)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

img1 = cv2.imread("karabin//orig.png", 0)
#img1 = cv2.resize(img1, (600, 800))
plt.imshow(img1),plt.show()

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
#des1 = des1.astype('float32')
good_matches=[]
avg_distance=[]
time_processing=[]
for i in range(1,121):
    img2 = cv2.imread(f"karabin//({i}).jpg", 0)
    #img2 = cv2.resize(img2, (600, 800))
    print(i)
    start_time=time.time()
    kp2, des2 = orb.detectAndCompute(img2, None)
        #des2 = des2.astype('float32')
    processing_time=time.time()-start_time

    if( len(kp2) <2 ):
        good_matches.append(0)
        time_processing.append(processing_time)
        avg_distance.append(np.nan)

    else:
        matches = flann.knnMatch(des1,des2,k=2)
            # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

            #Метрика 1 - количество хороших совпадений.
        count_good=0
        metric2=[]
            # ratio test as per Lowe's paper
        for i in range(len(matches)):
            if( len(matches[i])>1 ):
                if matches[i][0].distance < 0.7*matches[i][1].distance:
                    matchesMask[i]=[1,0]
                    count_good+=1
                    metric2.append( (matches[i][0].distance+matches[i][1].distance)/2)
        
        if( len(matches) == 0 ):
            metric1 = 0
        else:
            metric1=count_good/len(matches)
        good_matches.append(metric1)

        #Метрика 2. среднее расстояние между точками
        if( len(metric2) != 0):
            metric2=np.array(metric2).mean()
        else:
            metric2 = 0
        avg_distance.append(metric2)

        #Метрика 3. время обработки изображения.
        time_processing.append(processing_time)
    

import pandas as pd

data = pd.DataFrame()
data['good_matches'] = good_matches
data['avg_distance'] = avg_distance
data['time_processing'] = time_processing


plt.figure(figsize=(12,6))
plt.suptitle('карабин ORB')
plt.subplot(131)
plt.xlabel('good_matches')
plt.ylabel('N')
plt.hist(data.good_matches, bins=5, color = 'b')
plt.subplot(132)
plt.xlabel('avg_distance')
plt.hist(data.avg_distance, bins=5, color = 'g')
plt.subplot(133)
plt.xlabel('time_processing')
plt.hist(data.time_processing, bins=5, color = 'r')
plt.show()