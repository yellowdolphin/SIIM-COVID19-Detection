import os
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm 

image_source = '/kaggle/input/siim-covid19-resized-to-1024px-jpg'
os.makedirs('lung_crop', exist_ok=True)

if __name__ == '__main__':
    df = pd.read_csv('../../dataset/siim-covid19-detection/train_kfold.csv')

    for fold in range(5):
        tmp_df = df.loc[df['fold'] == fold]

        meles = []
        for _, row in tqdm(tmp_df.iterrows(), total=len(tmp_df)):
            #image_path = f"{image_source}/train/{row['imageid']}.jpg"
            ann_path = '../../dataset/lung_crop/labels/train/{}.xml'.format(row['imageid'])
            yolo_ann_path = '../../dataset/lung_crop/labels/train/{}.txt'.format(row['imageid'])
            
            tree = ET.parse(open(ann_path))
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            cnt = 0
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text

                xmlbox = obj.find('bndbox')
                x1, x2, y1, y2 = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
                cnt += 1
            assert cnt == 1

            xc = 0.5*(x1+x2)/width
            yc = 0.5*(y1+y2)/height
            w = (x2-x1)/width
            h = (y2-y1)/height

            with open(yolo_ann_path, 'w') as yolo_label_file:
                yolo_label_file.write('0 {} {} {} {}\n'.format(xc, yc, w, h))
                
    for fold in range(5):
        val_df = df.loc[df['fold'] == fold].sample(frac=1).reset_index(drop=True)
        train_df = df.loc[df['fold'] != fold].sample(frac=1).reset_index(drop=True)
        
        with open("../../dataset/lung_crop/yolov5_train_fold{}.txt".format(fold), "w") as yv5_tf:
            for _, row in train_df.iterrows():
                #image_path = '../../dataset/lung_crop/images/train/{}.png'.format(row['imageid'])
                image_path = f"{image_source}/train/{row['imageid']}.jpg"
                yv5_tf.write(image_path + '\n')

        with open("../../dataset/lung_crop/yolov5_valid_fold{}.txt".format(fold), "w") as yv5_vf:
            for _, row in val_df.iterrows():
                #image_path = '../../dataset/lung_crop/images/train/{}.png'.format(row['imageid'])
                image_path = f"{image_source}/train/{row['imageid']}.jpg"
                yv5_vf.write(image_path + '\n')
