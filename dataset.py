import os
from src import utils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader



def default_transforms():
    return transforms.Compose([
                                utils.PointSampler(1024),
                                utils.Normalize(),
                                utils.ToTensor()
                              ])

class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, foldername="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir,dir))]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = os.path.join(root_dir,category,foldername)
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pc_path'] = os.path.join(new_dir,file)
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = utils.read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pc_path = self.files[idx]['pc_path']
        category = self.files[idx]['category']
        with open(pc_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud,
                'category': self.classes[category]}




