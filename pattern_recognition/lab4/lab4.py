import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from joblib import dump,load
import time

image = 'hryvnia'
desk = 'orb'
init_desk = cv2.ORB_create()
classif = 'Logistic Regression"'
movie_name='IMG_5360.MOV'
original = cv2.imread('originals/hryvnia.png', 0)
replace =  cv2.imread('100.jpg')   # на что хотим заменять

MIN_MATCH_COUNT = 10  # минимальное кол-во точек, что совпали на тестовом изображении и на оригинале 

FLANN_INDEX_LSH= 6
index_params= dict(algorithm = FLANN_INDEX_LSH,table_number = 6,key_size = 12,multi_probe_level = 1)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

# куча кода с прошлой лабы ( инициализирем наши классификаторы ) ===================================================================
# ===============================================================================================================================================
des_list=['orb','sift','brisk']
kmeans=dict()
for i in des_list:
    kmeans[i]=load(f'written_models/{i}_kmeans.joblib')
logreg=dict()
svm=dict()
randomforest=dict()
for i in des_list:
    logreg[i]=load(f'written_models/{i}_logreg.joblib')
    svm[i]=load(f'written_models/{i}_svm.joblib')
    randomforest[i]=load(f'written_models/{i}_randomforest.joblib')
orb=cv2.ORB_create()
sift=cv2.SIFT_create(nfeatures=500)
brisk=cv2.BRISK_create()

descriptor=dict()
descriptor['orb']=orb
descriptor['sift']=sift
descriptor['brisk']=brisk

replace2 = replace
replace=cv2.cvtColor(replace, cv2.COLOR_BGR2GRAY)

def create_bag(labels_list):
    res=np.zeros((10*4,))
    for i in labels_list:
        res[i]+=1
    return res

def preprocess_image(img,estimator):
    img=cv2.resize( img, (img.shape[0]//4,img.shape[1]//4) )
    kp,des=descriptor[estimator].detectAndCompute(img,None)
    if(des is None):
        return None
    des=des.astype('float32')
    if(des.shape[0]>500):
        des=des[:500,:]
    kmeans_labels=kmeans[estimator].predict(des)
    bag_of_words=create_bag(kmeans_labels)
    return bag_of_words
# ===============================================================================================================================================
# ===============================================================================================================================================


cap=cv2.VideoCapture(movie_name)
print(classif)

fps = []
count = 0
img = replace
img1 = original
kp1, des1 = init_desk.detectAndCompute(img1,None)
start = time.time()

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame2 = frame
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        vect=dict()

        vect[desk]=preprocess_image(frame,desk)


        temp=vect[desk].reshape((1,vect[desk].shape[0]))
        pred=logreg[desk].predict(temp)[0]

        print( pred)


        if( pred == image):


            kp2, des2 = init_desk.detectAndCompute(frame,None)
            matches = flann.knnMatch(des1,des2,k=2)

            # находим совпадения точек
            good = []

            for i in matches:
                if( len(i) > 1):
                    m = i[0]
                    n = i[1]
                    if m.distance < 0.7*n.distance:
                        good.append(m)


            if len(good)>MIN_MATCH_COUNT: # строим линейное переобразование точек с оригинальной картинки на тесввую 
                                          # для того, что бы найти расположение 10 грн ( например ) на тестовой картинке
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = img1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                h,w= img.shape          # строим переобзование точек со 100 грн на 10 грн
                pts2 = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                matrix = cv2.getPerspectiveTransform(pts2, dst)
                rows,cols = frame.shape
                result = cv2.warpPerspective(replace2, matrix, (cols, rows)) 
                result = cv2.addWeighted(result, 0.8, frame2, 0.5, 1) # склеиваем картинки 

                cv2.imshow('Frame', result)
                #cv2.waitKey(0)
                
            else:
                print( "KUSOOOO Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT) )
                matchesMask = None
                cv2.imshow('Frame', frame2)
                #cv2.waitKey(0)
        else:            
            cv2.imshow('Frame', frame2)
            #cv2.waitKey(0)
    else:
        break
    count +=1  # для вывода фпс
    if( time.time()- start > 1): 
        fps.append(count)
        count = 0
        start = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print('mean fps', np.array(fps).mean())







