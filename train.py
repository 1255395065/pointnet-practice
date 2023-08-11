from tqdm import tqdm
import argparse
import os
import torch
from src import model
import dataset
from src import utils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)



def train(args):

    folders = [dir for dir in sorted(os.listdir(args.root_dir)) if os.path.isdir( os.path.join(args.root_dir,dir) )]
    classes = {folder: i for i, folder in enumerate(folders)};
    
    train_transforms = transforms.Compose([
        utils.PointSampler(1024),
        utils.Normalize(),
        utils.RandRotation_z(),
        utils.RandomNoise(),
        utils.ToTensor()
    ])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pointnet = model.PointNet(classes = 5)
    pointnet.to(device)
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=args.lr)
    
    train_ds = dataset.PointCloudData(args.root_dir, transform=train_transforms)
    #valid_ds = dataset.PointCloudData(args.root_dir, valid=True, foldername='test', transform=train_transforms)
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    #valid_loader = DataLoader(dataset=valid_ds, batch_size=args.batch_size*2)
    epoch_len = len(train_loader)
    
    try:
        os.mkdir(args.save_model_path)
    except OSError as error:
        print(error)
    
    print('Start training')
    for epoch in range(args.epochs):
        pointnet.train()
        running_loss = 0.0

        with tqdm(total=epoch_len,desc=f'Training: {epoch + 1}/{args.epochs}',postfix=dict,mininterval=0.3) as pbar:
            for data in train_loader:
                inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                optimizer.zero_grad()
                outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))

                loss = pointnetloss(outputs, labels, m3x3, m64x64)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(**{'loss'  : running_loss ,'lr'    : lr})
                pbar.update(1)
        
        #pointnet.eval()
        #correct = total = 0
        
        ## validation
        #if valid_loader:
        #    with torch.no_grad():
        #        for data in valid_loader:
        #            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
        #            outputs, __, __ = pointnet(inputs.transpose(1,2))
        #            _, predicted = torch.max(outputs.data, 1)
        #            total += labels.size(0)
        #            correct += (predicted == labels).sum().item()
        #    val_acc = 100. * correct / total
        #    print('Valid accuracy: %d %%' % val_acc)

        # save the model  
        if epoch % 50 == 0:
            saveroot = os.path.join(args.save_model_path,'model_')+str(epoch)+'.pth'
            torch.save(pointnet.state_dict(), saveroot)
            print('Model saved to ', saveroot)


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--root_dir', default='./ModelNet10/', type=str,
                            help='dataset directory')
    parser.add_argument('--batch_size', default=32, type=int,
                            help='training batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                            help='learning rate')
    parser.add_argument('--epochs', default=300, type=int,
                            help='number of training epochs')
    parser.add_argument('--save_model_path', default='./savemodel/', type=str,
                            help='checkpoints dir')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train(args)




