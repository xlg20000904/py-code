# -*- coding: utf-8 -*-
import os
import cv2
import torch.utils.data as data
import pandas as pd
import random
from torchvision import transforms
from utils import *

class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel', args.label_path), sep=' ', header=None)
        
        name_c = 0
        label_c = 1
        if phase == 'train':
            # ：如果是训练阶段，这一行代码从 DataFrame df 中选择了所有图像文件名以 'train' 开头的行，
            # 并将它们存储在名为 dataset 的 DataFrame 中。这就构建了训练集的子集。
            dataset = df[df[name_c].str.startswith('train')]
        else:
            df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
            dataset = df[df[name_c].str.startswith('test')]
            #     这一行代码从 dataset 中提取标签列（索引为 label_c），并将它们存储在 self.label 中。
            #     值减去 1 可能是为了将标签从 1-based 编码转换为 0-based 编码。
        self.label = dataset.iloc[:, label_c].values - 1
        # 这一行代码从
        # dataset
        # 中提取图像文件名列（索引为
        # name_c）并将它们存储在
        # images_names
        # 中。
        images_names = dataset.iloc[:, name_c].values
        # 这一行代码创建了一个包含两个函数 flip_image 和 add_g 的列表，
        # 这些函数可能用于数据增强，但代码片段中未显示其具体实现。
        self.aug_func = [flip_image, add_g]
        self.file_paths = []
        # 这一行代码将 self.clean 设置为一个布尔值，用于表示是否使用了一个名为 'list_patition_label.txt' 的标签文件。
        # 如果 args.label_path 等于 'list_patition_label.txt'，则 self.clean 为 True，否则为 False。
        self.clean = (args.label_path == 'list_patition_label.txt')
        
        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(file_name)

    # 返回文件路径的长度
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        # print(self.file_paths[idx])
        # 从文件路径列表
        # self.file_paths
        # 中读取图像文件，使用OpenCV的
        # imread
        # 函数读取图像。
        image = cv2.imread(self.file_paths[idx])
        # 这一行代码翻转了图像的通道顺序，将BGR通道顺序转换为RGB通道顺序。
        # OpenCV通常将图像加载为BGR通道顺序，
        # 而许多深度学习框架（如PyTorch和TensorFlow）使用RGB通道顺序。
        image = image[:, :, ::-1]
        
        
        if not self.clean:
            # 创建了一个新的图像变量 image1 并将其设置为与原始图像 image 相同
            image1 = image
            # self.aug_func[0](image) 通过调用列表 self.aug_func 中的第一个数据增强函数来增强图像 image1。

            image1 = self.aug_func[0](image)
            # 将增强后的图像
            # image1
            # 进一步进行数据转换（假设
            # self.transform
            # 是一个数据转换函数，可能用于缩放、标准化等操作）。
            image1 = self.transform(image1)
        # 对原始图像进行第二种数据增强操作
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                # 检查是否定义了数据转换函数,如果定义了,则将函数进行最终的数据转换
                image = self.aug_func[1](image)

        if self.transform is not None:
            # 它使用transforms库中的RandomHorizontalFlip函数来进行水平翻转操作。
            # p=1参数表示水平翻转的概率为100%
            image = self.transform(image)
        
        if self.clean:
            image1 = transforms.RandomHorizontalFlip(p=1)(image)

        # 回一个元组，包含以下四个元素：
        # image：原始或经过数据增强和转换后的图像。
        # label：与图像相关联的标签。
        # idx：样本的索引。
        # image1：如果数据不是干净的，则包含数据增强和转换的第一个版本的图像。如果数据干净，这个值将是通过随机水平翻转生成的。
        return image, label, idx, image1