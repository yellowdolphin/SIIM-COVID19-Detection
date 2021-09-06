import os
from pathlib import Path
import pandas as pd 
import numpy as np
from tqdm import tqdm

image_source = '/kaggle/input/chexpert-v10-small/CheXpert-v1.0-small'
padchest_source = '/kaggle/input/padchest/images-224'

if __name__ == '__main__':
    ### remove unused file in chexpert dataset
    # Avoid copy of image_sorce and sym links (inefficient), update csv files instead.
    all_files = []
    for split in ['train', 'valid']:
        df = pd.read_csv(f'../../dataset/external_dataset/ext_csv/chexpert_{split}.csv')
        df['image_path'] = df.image_path.str.replace('../../dataset/external_dataset/chexpert', image_source)

        print(f"CheXpert {split}: searching for images in {image_source} ...")
        file_exists = df.image_path.map(os.path.exists)

        print(f"    found {sum(file_exists)} / {len(chexpert_df)} images")
        df.loc[file_exists].to_csv(csv_file)
        print(f"    updated {csv_file}")
        all_files.append(df.image_path.values)

    print("    testing for train/valid duplicates ...")
    all_files = np.concatenate(all_files)
    assert len(all_files) == len(set(all_files)), f'{len(all_files) - len(set(all_files))} duplicates in train + valid'
    del all_files
    print("    all done")
    
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
    # Avoid copy of image_sorce and sym links (inefficient), update csv file instead.
    padchest_df = pd.read_csv('../../dataset/external_dataset/ext_csv/padchest.csv').drop_duplicates()
    padchest_df['image_path'] = padchest_df.image_path.str.replace('../../dataset/external_dataset/padchest/images', padchest_source)
    file_exists = padchest_df.image_path.map(os.path.exists)
    print(f"PadChest: found {sum(file_exists)} / {len(padchest_df)} images")
    new_csv_file = '../../dataset/external_dataset/ext_csv/padchest.csv'
    padchest_df.to_csv(new_csv_file)
    print(f"          wrote {new_csv_file}")
    print(f"          all done")