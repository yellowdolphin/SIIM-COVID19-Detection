import numpy as np
import pandas as pd 
from tqdm import tqdm

image_source = '/kaggle/input/vinbigdata-chest-xray-resized-png-1024x1024'

if __name__ == '__main__':
    train_df = pd.read_csv('/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv')

    output = []
    for image_id, grp in tqdm(train_df.groupby('image_id')):
        image_path = f'{image_source}/train/{image_id}.png'
        nf_cnt = 0
        for rad_id, grp1 in grp.groupby('rad_id'):
            if 'No finding' in grp1.class_name.values:
                assert len(grp1) == 1
                nf_cnt += 1
        if nf_cnt > 1:
            normal = 1
            pneumonia = 0
            label = 'none 1 0 0 1 1'
            hasbox = 0
        else:
            normal = -1
            pneumonia = -1
            label = 'unknown'
            hasbox = 0
        output.append([image_path, normal, pneumonia, label, hasbox])
    new_df = pd.DataFrame(data=np.array(output), columns=['image_path', 'normal', 'pneumonia', 'label', 'hasbox'])
    new_df.to_csv('../../dataset/external_dataset/ext_csv/vin.csv', index=False)
    print(new_df.shape)
    