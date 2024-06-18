import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader,Dataset
import numpy as np

import resnet

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm



def main():
    
    BATCH_SIZE = 4

    device = torch.device('cuda:0')

    dataset = emuData()
    dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)

    sample = dataset.__getitem__(0)[0]

    backbone = resnet.ResNet18Gray()
    projector = LinearProjector(in_dim=512,out_dim=512)
    probe = LiRPA_Probe()
    model = torch.nn.Sequential(backbone,projector,probe)

    model=model.to(device)

    model_bd = BoundedModule(model,sample.unsqueeze(0))

    ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)

    model_bd.eval()

    with torch.no_grad():

        for aud, label in dataloader:
            aud, label = aud.to(device), label.to(device)

            aud_ptbd = BoundedTensor(aud,ptb)

            pred = model_bd(aud_ptbd)

            lb_a,ub_a = model_bd.compute_bounds(x=(aud_ptbd,)) #error occurs here

            print(f"{lb_a},{ub_a}\n")

class LinearProjector(nn.Module):
    def __init__(self, in_dim=512, out_dim=2048, frames=25):
        super().__init__()
        self.frames = frames
        self.net = nn.Linear(in_dim, out_dim)
        self.norm_layer = nn.BatchNorm1d(out_dim)

    def forward(self, x:torch.Tensor):
        # x = x.unsqueeze(-1).repeat(1,1,25)
        if len(x.shape)<3:
            x=x.unsqueeze(-1)
        n = self.frames
        x = x.transpose(1, 2).contiguous().view(-1, x.size(1))
        x = self.net(x)
        x = self.norm_layer(x)
        x = x.view(-1, n, x.size(-1))  # keep this for transformer
        return x

class LiRPA_Probe(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128):
        super().__init__()
        norm_layer = nn.BatchNorm1d
        self.layer1 = self.make_mlp_layer(in_dim, hidden_dim, norm_layer=norm_layer)
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def make_mlp_layer(self,in_dim, out_dim=4096, relu=True, norm_layer=nn.BatchNorm1d):
        layers = [nn.Linear(in_dim, out_dim)]
        if norm_layer:
            layers.append(norm_layer(out_dim))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        n = x.size(1)
        x = x.reshape((-1, x.size(-1)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape((-1, n, x.size(-1)))
        x = x.mean(1)
        return self.sigmoid(x)

class emuData(Dataset):
    def __init__(self) -> None:
        super().__init__()
    def __len__(self):
        return 1024
    def __getitem__(self,idx):
        tensor1 = torch.rand(1,100,80)*2-1
        tensor2 = torch.randint(0,2,(1,)).float()
        return tensor1,tensor2

main()
