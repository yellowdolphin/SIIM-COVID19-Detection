import os
import pandas as pd 
import numpy as np
from tqdm import tqdm

image_source = '/kaggle/input/chexpert-v10-small'
padchest_source = '/kaggle/input/padchest/images-224'

if __name__ == '__main__':
    ### remove unused file in chexpert dataset
    chextpert_train_df = pd.read_csv('../../dataset/external_dataset/ext_csv/chexpert_train.csv')
    chextpert_train_df['image_path'] = chextpert_train_df['image_path'].str.replace(
        '../../dataset/external_dataset/chexpert', image_source)
    chextpert_valid_df = pd.read_csv('../../dataset/external_dataset/ext_csv/chexpert_valid.csv')
    chextpert_valid_df['image_path'] = chextpert_valid_df['image_path'].str.replace(
        '../../dataset/external_dataset/chexpert', image_source)
    chextpert_df = pd.concat([chextpert_train_df, chextpert_valid_df], ignore_index=False)
    useful_image_paths = np.unique(chextpert_df.image_path.values)
    image_paths = []
    #for rdir, _, files in os.walk(f'../../dataset/external_dataset/chexpert/train'):
    for rdir, _, files in os.walk(image_source):   # or should only /train be included?
        for file in files:
            image_path = os.path.join(rdir, file)
            if os.path.isfile(image_path):
                image_paths.append(image_path)
    print(len(chextpert_df), len(image_paths))
    for image_path in tqdm(image_paths):
        if image_path not in useful_image_paths:
            os.remove(image_path)
            print('remove {} ...'.format(image_path))
    
    ### remove unused file in chest14 dataset
    # I don't use chest14 for now.
    #chest14_train_df = pd.read_csv('../../dataset/external_dataset/ext_csv/chest14_train.csv')
    #chest14_valid_df = pd.read_csv('../../dataset/external_dataset/ext_csv/chest14_valid.csv')
    #chest14_df = pd.concat([chest14_train_df, chest14_valid_df], ignore_index=False)
    #useful_image_paths = []
    #for image_path in np.unique(chest14_df.image_path.values):
    #    useful_image_paths.append('../../dataset/external_dataset/chest14/images/{}'.format(image_path))
    #image_paths = []
    #for rdir, _, files in os.walk('../../dataset/external_dataset/chest14/images'):
    #    for file in files:
    #        image_path = os.path.join(rdir, file)
    #        if os.path.isfile(image_path):
    #            image_paths.append(image_path)
    #print(len(chest14_df), len(image_paths))
    #for image_path in tqdm(image_paths):
    #    if image_path not in useful_image_paths:
    #        os.remove(image_path)
    #        print('remove {} ...'.format(image_path))
    
    ### remove unused and missing files in padchest dataset
    padchest_df = pd.read_csv('../../dataset/external_dataset/ext_csv/padchest.csv').drop_duplicates()
    padchest_df['source_path'] = padchest_df.image_path.str.replace('../../dataset/external_dataset/padchest/images', padchest_source)
    file_exists = padchest_df.source_path.map(os.path.exists)
    print(f"PadChest: found {sum(file_exists)} / {len(padchest_df)} images")
    print(f"          creating symb links in dataset/external_dataset/padchest/images ...")
    for index, (src, target) in padchest_df.loc[file_exists, ['source_path', 'image_path']].iterrows():
        os.symlink(src, target)