import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from shutil import copy
from sklearn.model_selection import train_test_split

data = 'train_data/train.csv'
images = 'train_data/train'
output = 'train_data/data'


def pre_process(df, data_type='train'):
    label = 0
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        yolo_format = []
        image_id = row['image_id']
        for bbox in row['bboxes']:
            xmin = bbox[0]
            ymin = bbox[1]
            width = bbox[2]
            height = bbox[3]

            x_center = xmin + width / 2
            y_center = ymin + height / 2

            # normalization as mentioned in the YoloV5 doc
            x_center /= 1024
            width /= 1024
            y_center /= 1024
            height /= 1024

            yolo_format.append([label, x_center, y_center, width, height])

        np.savetxt(f"{os.path.join(output, data_type, 'labels', image_id)}.txt", yolo_format,
                   fmt=["%d", "%f", "%f", "%f", '%f'])
        copy(f"{os.path.join(images, image_id)}.jpg", f"{os.path.join(output, data_type, 'images')}")


if __name__ == "__main__":
    df = pd.read_csv(data)
    df['bbox'] = df['bbox'].apply(eval)
    df = df.groupby('image_id')['bbox'].apply(list).reset_index(name='bboxes')
    train, validation = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    pre_process(train)
    pre_process(validation, 'val')
