from utils.util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from datetime import datetime
import logging
from tqdm import tqdm
import argparse

# training configuration
split = 1
iterations = 2000
it_save = 100  # save model every 100 iterations
n_cpu = 8 # 8
seq_length = 64
k = 10  # frozen layers

training_mode = 1
'''
0: basic mode(image)

1: norm1+2Dpose
2: norm1+2Dpose+conf 
3: norm2+2Dpose
4: norm2+2Dpose+conf 

5: 1 + resblock
6: 2 + resblock
7: 3 + resblock
8: 4 + resblock

9: 3Dpose
'''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_mode', type=int, default=1, help='input video')
    args = parser.parse_args()
    training_mode = args.training_mode
    
    if training_mode==0:
        from dataloaders.dataloader import GolfDB, Normalize, ToTensor
        from networks.model import EventDetector
        save_model_name = 'image'
        bs = 12  # batch size       # 22 -> 16 -> 12
    elif training_mode==1:
        from dataloaders.dataloader_with_pose1 import GolfDB, Normalize, ToTensor
        from networks.model1 import EventDetector
        save_model_name = 'pose1'
        pose_dir='data/poses_160/'
        bs = 128  # batch size       # 22 -> 16 -> 12
    elif training_mode==2:
        from dataloaders.dataloader_with_pose2 import GolfDB, Normalize, ToTensor
        from networks.model2 import EventDetector
        save_model_name = 'pose2'
        pose_dir='data/poses_160/'
        bs = 128  # batch size       # 22 -> 16 -> 12
    elif training_mode==3:
        from dataloaders.dataloader_with_pose3 import GolfDB, Normalize, ToTensor
        from networks.model1 import EventDetector
        save_model_name = 'pose3'
        pose_dir='data/poses_160/'
        bs = 128  # batch size       # 22 -> 16 -> 12
    elif training_mode==4:
        from dataloaders.dataloader_with_pose4 import GolfDB, Normalize, ToTensor
        from networks.model2 import EventDetector
        save_model_name = 'pose4'
        pose_dir='data/poses_160/'
        bs = 128  
    elif training_mode==5:
        from dataloaders.dataloader_with_pose1 import GolfDB, Normalize, ToTensor
        from networks.model1_1 import EventDetector
        save_model_name = 'pose1_1'
        pose_dir='data/poses_160/'
        bs = 128  
    elif training_mode==6:
        from dataloaders.dataloader_with_pose2 import GolfDB, Normalize, ToTensor
        from networks.model2_1 import EventDetector
        save_model_name = 'pose2_1'
        pose_dir='data/poses_160/'
        bs = 128  
    elif training_mode==7:
        from dataloaders.dataloader_with_pose3 import GolfDB, Normalize, ToTensor
        from networks.model1_1 import EventDetector
        save_model_name = 'pose3_1'
        pose_dir='data/poses_160/'
        bs = 128  
    elif training_mode==8:
        from dataloaders.dataloader_with_pose4 import GolfDB, Normalize, ToTensor
        from networks.model2_1 import EventDetector
        save_model_name = 'pose4_1'
        pose_dir='data/poses_160/'
        bs = 128  
    else: 
        from dataloaders.dataloader_with_3Dpose import GolfDB, Normalize, ToTensor
        from networks.model3d import EventDetector
        save_model_name = 'pose_d3'
        pose_dir='data/poses3D_160/'
        bs = 128  
    
    now = datetime.now()
    os.makedirs('models', exist_ok=True)
    save_model_folder = 'models/{}'.format(save_model_name) # + now.strftime('%Y%m%d_%H%M')
    os.makedirs(save_model_folder, exist_ok=True)
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(save_model_folder, 'train.log'), level=logging.INFO)
    
    
    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    freeze_layers(k, model)
    model.train()
    model.cuda()

    if training_mode==0:
        dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=True)
    else:
        dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                        vid_dir='data/videos_160/',
                        pose_dir=pose_dir,
                        seq_length=seq_length,
                        transform=transforms.Compose([ToTensor(),
                                                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                        train=True)

    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    losses = AverageMeter()

    i = 0
    while i < iterations:
        for sample in tqdm(data_loader, 0):
            inputs, labels = sample['inputs'].cuda(), sample['labels'].cuda()
            logits = model(inputs)  
            labels = labels.view(bs*seq_length)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), inputs.size(0))
            optimizer.step()
            i += 1
            if i % 10 == 0:
                print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, save_model_folder+'/swingnet_{0:05d}.pth.tar'.format(i))
                logging.info('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            if i == iterations:
                break



