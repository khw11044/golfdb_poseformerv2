import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from utils.util import correct_preds
import glob 
from tqdm import tqdm

training_mode = 8
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
load_model_root = 'models/pose4_1_'

bs = 12  # batch size       # 22 -> 16 -> 12

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


def eval(model, split, seq_length, n_cpu, disp):
    if training_mode==0:
        dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
                        vid_dir='data/videos_160/',
                        seq_length=seq_length,
                        transform=transforms.Compose([ToTensor(),
                                                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                        train=False)
    else:
        dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
                        vid_dir='data/videos_160/',
                        pose_dir=pose_dir,
                        seq_length=seq_length,
                        transform=transforms.Compose([ToTensor(),
                                                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                        train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []

    for i, sample in enumerate(tqdm(data_loader, 0)):
        inputs, labels = sample['inputs'], sample['labels']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < inputs.shape[1]:
            if (batch + 1) * seq_length > inputs.shape[1]:
                if training_mode==0:
                    input_batch = inputs[:, batch * seq_length:, :, :, :]
                else:
                    input_batch = inputs[:, batch * seq_length:, :]
            else:
                if training_mode==0:
                    input_batch = inputs[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
                else:
                    input_batch = inputs[:, batch * seq_length:(batch + 1) * seq_length, :]
                
            logits = model(input_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        _, _, _, _, c = correct_preds(probs, labels.squeeze())
        if disp:
            print(i, c)
        correct.append(c)
    PCE = np.mean(correct)
    return PCE


if __name__ == '__main__':

    split = 1
    seq_length = 64
    n_cpu = 8

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    load_model_name = sorted(glob.glob(load_model_root +'/*.tar'))[-1]
    save_dict = torch.load(load_model_name)
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    PCE = eval(model, split, seq_length, n_cpu, False)   # True 
    print('Average PCE: {}'.format(PCE))


