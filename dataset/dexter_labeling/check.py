import numpy as np
import cv2
import os
import pandas as pd


#Kithen
path = 'EgoDexter/data/Desk/color/'
df = pd.read_csv('bbox_egodexter_kitchen.csv')




files = os.listdir(path)
files = [path + x for x in files]
files = np.sort(files)




for i, file in enumerate(files, 0):
    print(file)
    frame = cv2.imread(file)
    
    #bbox = list(df.iloc[i][['x','y','w','h']])
    #print(str(df.iloc[i][['img_name']]))
    #print()

 #   p1 = (int(bbox[0]), int(bbox[1]))
 #   p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
 #   cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        
    cv2.imshow('frame',frame)
    cv2.waitKey(0)


cv2.destroyAllWindows()

