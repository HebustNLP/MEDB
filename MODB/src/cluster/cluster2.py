import csv
from scipy import stats
import torch
import numpy as np

from . import cluster3 as new_GBNR


def calculate_distances(center, p):
    return ((center - p) ** 2).sum(axis=0) ** 0.5


class GBNR:
    @classmethod
    def forward(self,args, input_,select):

        self.batch_size = input_.size(0)
        input_main = input_[:, 3:]  # noise_label+64 [bs,65]
        self.input = input_[:, 4:]
        self.res = input_[:, 1:2]
        self.index = input_[:, 2:3]
        pur = input_[:, 0].cpu().numpy().tolist()[0]

        self.flag = 0

        numbers,  result, center, radius = new_GBNR.main(args,input_main,select)


        labels=[]
        centroids=[]
        for gb in center:
            label=  gb[0]
            centroid=gb[1:]
            labels.append(label)
            centroids.append(centroid)
        self.labels  = torch.Tensor(labels)
        self.centroids=  torch.Tensor(centroids)
        radius=  torch.Tensor(radius)


        return  self.centroids, self.labels, radius,
