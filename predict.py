import os
import torch
from src import model
from src import utils
from torchvision import transforms


def main():
    classes = 5
    class_list = ['bathtub', 'desk', 'dresser', 'night_stand', 'table']
    weights_path = "./model_100.pth"
    pc_path = "./bathtub_0107.off"

    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    pointnet = model.PointNet(classes = 5)
    pointnet.load_state_dict(torch.load(weights_path, map_location='cpu'))
    pointnet.to(device)

    data_transform = transforms.Compose([
                                utils.PointSampler(1024),
                                utils.Normalize(),
                                utils.ToTensor()
                              ])

    with open(pc_path, 'r') as f:
        verts, faces = utils.read_off(f)
    pointcloud = data_transform((verts, faces))

    pointnet.eval()  
    with torch.no_grad():
         pointcloud.to(device)
         pointcloud = pointcloud.unsqueeze(dim = 0)
         outputs, _, _ = pointnet(pointcloud.transpose(1,2).float())
         outputs = outputs.squeeze(dim = 0)
         idx = torch.argmax(outputs)
         print(class_list[idx])

if __name__ == '__main__':
    main()






