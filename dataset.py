from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import os
from PIL import Image

class CustomNpzFolderLoader(Dataset):
    def __init__(self, npz_path, labels, data_transforms=None):
        super(CustomNpzFolderLoader, self).__init__()
        self.npz_path = npz_path
        self.labels = labels
        # self.dataset = dataset
        self.data_transforms = data_transforms

    def __getitem__(self, item):
        feature = self.npz_path[item]
        # feature0=Image.fromarray(np.uint8(feature[0,:,:]))
        # feature1=Image.fromarray(np.uint8(feature[1,:,:]))
        # feature2=Image.fromarray(np.uint8(feature[2,:,:]))
        # feature=np.stack((feature0,feature1,feature2))
        label = self.labels[item]
        if self.data_transforms is  None:
            # feature = self.data_transforms[self.dataset](feature)
            feature = self.data_transforms(feature)
            # feature0 = self.data_transforms(feature0)
            # feature1 = self.data_transforms(feature1)
            # feature2 = self.data_transforms(feature2)
            # feature=np.stack((feature0[0,:,:],feature1[0,:,:],feature2[0,:,:]))
            
        return feature, label

    def __len__(self):
        return len(self.labels)

class CustomNpzFolderLoader_test(Dataset):
    def __init__(self, npz_path, data_transforms=None):
        super(CustomNpzFolderLoader_test, self).__init__()
        self.npz_path = npz_path

        # self.dataset = dataset
        self.data_transforms = data_transforms

    def __getitem__(self, item):
        feature = self.npz_path[item]
        if self.data_transforms is not None:
            # feature = self.data_transforms[self.dataset](feature)
            feature = self.data_transforms(feature)
        return feature

    def __len__(self):
        return len(self.npz_path)