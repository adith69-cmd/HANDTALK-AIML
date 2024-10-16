import os
import cv2
cap=cv2.VideoCapture(0)
directory='Image/'
while True:
    _,frame=cap.read()
    count = {
             'hii': len(os.listdir(directory+"/hii")),
             'thanks': len(os.listdir(directory+"/thanks")),
             'noo': len(os.listdir(directory+"/noo")),
             }
    # cv2.putText(frame, "a : "+str(count['a']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "b : "+str(count['b']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "c : "+str(count['c']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    
    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame,(0,40),(300,400),(255,255,255),2)
    cv2.imshow("data",frame)
    cv2.imshow("ROI",frame[40:400,0:300])
    frame=frame[40:400,0:300]
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'hii/'+str(count['hii'])+'.png',frame)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory+'thanks/'+str(count['thanks'])+'.png',frame)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'noo/'+str(count['noo'])+'.png',frame)

cap.release()
cv2.destroyAllWindows()