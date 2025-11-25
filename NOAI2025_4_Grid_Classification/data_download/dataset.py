import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import random
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, train_file, mode=None, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.mode = mode
        with open(train_file, "r", encoding="gbk") as file: 
            for i,line in enumerate(file):
                line = line.strip()
                if not line:
                    continue
                    
                # 解析txt文件
                fields = [field.strip() for field in line.split(',')]
                if len(fields) != 4 and self.mode == "train":
                    raise ValueError("Each line must have 4 fields: id,img_url,label,class")
                if self.mode == "train":
                    img_name, img_url, label_str, class_name = fields
                    label = int(label_str)
                    # print(i,img_name, img_url, label, class_name)
                    
                    # 获取本地文件
                    try:
                        img_path = os.path.join(data_dir, f'{img_name}.jpg')
                        image = Image.open(img_path).convert('RGB')
                    except:
                        img_path = os.path.join(data_dir, f'{img_name}.png')
                        image = Image.open(img_path).convert('RGB')

                    self.images.append(image)
                    self.labels.append(label)
                elif self.mode == "val" or self.mode == "test":
                    img_name, img_url, class_name = fields
                    # print(i,img_name, img_url, label, class_name)
                    # 获取本地文件
                    try:
                        img_path = os.path.join(data_dir, f'{img_name}.jpg')
                        image = Image.open(img_path).convert('RGB')
                    except:
                        img_path = os.path.join(data_dir, f'{img_name}.png')
                        image = Image.open(img_path).convert('RGB')

                    self.images.append(image)
                else:
                    raise ValueError("Invalid mode")
        
        ########################CODE HERE###########################
        '''
        请编写数据增广函数
        Please implement the data augmendation function
        '''
        def augmentation(images, labels, augmentation_sample_num=400):
            # 在这里编写代码
            # code here
            neg_images = []

            for (img, label) in zip(images, labels):
                if label == 0:
                    neg_images.append(img)
            
            random.shuffle(neg_images)

            augmentation_images, augmentation_labels = [], []
            COL = 2 #指定拼接图片的列数/set the number of columns for image concatenation
            ROW = 2 #指定拼接图片的行数/set the number of rows for image concatenation
            UNIT_HEIGHT_SIZE = 128 #宫格单元图片高度/height of unit image
            UNIT_WIDTH_SIZE = 128 #宫格单元图片宽度/width of unit image
            for i in range(0,len(neg_images)//4,COL*ROW):
                augmentation_image = Image.new('RGB', (UNIT_WIDTH_SIZE * COL, UNIT_HEIGHT_SIZE * ROW))
                for row in range(ROW):
                    for col in range(COL):
                        augmentation_image.paste(neg_images[i*COL*ROW+COL*row+col], (UNIT_WIDTH_SIZE*col, UNIT_HEIGHT_SIZE*row))
                augmentation_images.append(augmentation_image)
                augmentation_labels.append(1)
            return augmentation_images, augmentation_labels


        ########################CODE HERE#############################
        if self.mode == "train":
            augmentation_images, augmentation_labels = augmentation(self.images, self.labels)
            self.images.extend(augmentation_images)
            self.labels.extend(augmentation_labels)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        if self.mode == "train":
            label = self.labels[idx]
            return image, label
        else:
            return image