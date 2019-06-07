mport numpy as np
import cv2
import os
import pandas as pd


data_folder = 'EgoDexter/data/'
seq_folder = 'Desk/color/'


files = os.listdir(data_folder + seq_folder)

filenames = [seq_folder + x for x in files]
filenames = np.sort(filenames)

files = [data_folder + seq_folder + x for x in files]
files = np.sort(files)



bbox_data = {'img_name': [],
             'x': [],
             'y': [],
             'w': [],
             'h': []}

tracker = cv2.TrackerKCF_create()

start_num = 0
end_num = len(files)

frame = cv2.imread(files[start_num])
bbox = cv2.selectROI(frame, False)

bbox_data['img_name'].append(filenames[start_num])
bbox_data['x'].append(bbox[0])
bbox_data['y'].append(bbox[1])
bbox_data['w'].append(bbox[2])
bbox_data['h'].append(bbox[3])

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

for i, file in enumerate(files[(start_num+1):end_num], start_num+1):
    print(file)
    frame = cv2.imread(file)
    
    ok, bbox = tracker.update(frame)
    
    bbox_data['img_name'].append(filenames[i])
    bbox_data['x'].append(bbox[0])
    bbox_data['y'].append(bbox[1])
    bbox_data['w'].append(bbox[2])
    bbox_data['h'].append(bbox[3])
    
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        

        
    cv2.imshow('frame',frame)
    cv2.waitKey(0)


cv2.destroyAllWindows()


df = pd.DataFrame(bbox_data)
df = df[['img_name','x','y','w','h']]
df.to_csv('bbox_data_egodexter.csv', index = False)

