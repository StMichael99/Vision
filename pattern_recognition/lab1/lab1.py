import cv2

# open video
cap = cv2.VideoCapture(0)

# for saving
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# stream
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:

    cv2.imshow('Frame1',frame)
    out.write(frame)

    # if you want to stop, you have to press q 
    if cv2.waitKey(1) & 0xFF == ord('q'):
      print ('You stop reccording')
      break
  else: 
    break
cap.release()
out.release()



cap = cv2.VideoCapture('output.avi')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        # convert picture to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # make picture 3 dimesion ( it is for colored rectangle )
        grayBGR= cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # drow red rectangle
        start_point = (5, 5) 
        end_point = (220, 220) 
        color = (0, 0, 255) 
        thickness = 2
        cv2.rectangle(grayBGR, start_point, end_point, color, thickness)

        # drow green line
        (x1, y1) = (300, 300 )
        (x2, y2) = (340, 340 )
        cv2.line(grayBGR, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        cv2.imshow('frame 2',grayBGR)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

# job is finished
cv2.destroyAllWindows()
