import sys
import argparse
import cv2
import os 
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
from IPython import embed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from eval import ToTensor, Normalize

sys.path.append(os.getcwd())
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
from common.model_poseformer import PoseTransformerV2 as Model
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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

'''
if training_mode==1:
    from networks.model1 import EventDetector
    save_model_name = 'pose1'
elif training_mode==2:
    from networks.model2 import EventDetector
    save_model_name = 'pose2'
elif training_mode==3:
    from networks.model1 import EventDetector
    save_model_name = 'pose3'   
elif training_mode==4:
    from networks.model2 import EventDetector
    save_model_name = 'pose4'
elif training_mode==5:
    from networks.model1_1 import EventDetector
    save_model_name = 'pose1_1'
elif training_mode==6:
    from networks.model2_1 import EventDetector
    save_model_name = 'pose2_1'
elif training_mode==7:
    from networks.model1_1 import EventDetector
    save_model_name = 'pose3_1'
elif training_mode==8:
    from networks.model2_1 import EventDetector
    save_model_name = 'pose4_1'
else: 
    from networks.model3d import EventDetector
    save_model_name = 'pose_d3'


class SampleVideo(Dataset):
    def __init__(self, poses, width, height, mode, transform=None):
        self.poses = poses
        self.width = width
        self.height = height
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # preprocess and return frames
        poses2D = self.poses
        if self.mode==1 or self.mode==5:
            normalize_coor_pose = normalize_screen_coordinates(poses[:,:,:2], self.width, self.height).reshape(-1,2,17).reshape(-1,34)
        elif self.mode==2 or self.mode==6:
            normalize_coor_pose = normalize_screen_coordinates(poses[:,:,:2], self.width, self.height).reshape(-1,2,17)
            normalize_coor_pose = np.concatenate((normalize_coor_pose, poses[:,:,-1:].reshape(-1,1,17)),axis=1).reshape(-1,51)
        elif self.mode==3 or self.mode==7:
            normalize_coor_pose = canonical_normalize_coordinates(poses[:,:,:2])
        elif self.mode==4 or self.mode==8:
            normalize_coor_pose = canonical_normalize_coordinates(poses[:,:,:2])
            normalize_coor_pose = np.concatenate((normalize_coor_pose, poses[:,:,-1:].reshape(-1,17)),axis=1).reshape(-1,51)
        labels = np.zeros(len(poses2D)) # only for compatibility with transforms
        sample = {'inputs': normalize_coor_pose, 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    scores = scores.reshape(scores.shape[0],scores.shape[1],scores.shape[2],1)
    kps_conf = np.concatenate([keypoints, scores], axis=3)
    # re_kpts = revise_kpts(keypoints, scores, valid_frames)
    print('Generating 2D pose successful!')


    return kps_conf[0], width, height

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='test_video.mp4', help='input video')
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames to use per forward pass', default=64)
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--model_name', type=str, default='27_243_45.2.bin', help='input video')
    args = parser.parse_args()
    seq_length = args.seq_length
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './demo/output/' + video_name + '/'

    poses, width, height = get_pose2D(video_path, output_dir)
    
    ds = SampleVideo(poses, width, height, mode=training_mode, transform=transforms.Compose([ToTensor(),
                                Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])]))
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    
    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    load_model_root = 'models/{}'.format(save_model_name)
    load_model_name = sorted(glob.glob(load_model_root +'/*.tar'))[-1]
    try:
        save_dict = torch.load(load_model_name)
    except:
        print("Model weights not found. Download model weights and place in 'models' folder. See README for instructions")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")
    
    print('Testing...')
    for sample in dl:
        inputs = sample['inputs']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < inputs.shape[1]:
            if (batch + 1) * seq_length > inputs.shape[1]:
                input_batch = inputs[:, batch * seq_length:, :]
            else:
                input_batch = inputs[:, batch * seq_length:(batch + 1) * seq_length, :]
            logits = model(input_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    print('Predicted event frames: {}'.format(events))
    cap = cv2.VideoCapture(video_path)

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Condifence: {}'.format([np.round(c, 3) for c in confidence]))

    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        _, img = cap.read()
        cv2.putText(img, '{:.3f}'.format(confidence[i]), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        cv2.putText(img, '{}'.format(event_names[i]), (20, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        cv2.imshow(event_names[i], img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    print('Generating demo successful!')


