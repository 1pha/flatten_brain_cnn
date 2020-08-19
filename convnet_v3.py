import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim

class ConvNet(nn.Module):
    '''
    Tries to imitate ResNet18 Basic Building Blocks and that looks like
    Conv-bn-relu -Conv-bn-relu -MaxPool
    '''

    def __init__(self, ctype, in_channels=6, filters=None, flat_nodes=None, kernel_size=5, stride=1):
        super(ConvNet, self).__init__()

        if filters is None:
            filters = [16, 32, 64, 128]

        self.ctype = ctype
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.filters = filters


        if self.ctype=='binary':
            out_node = 1
            self.last_layer = F.sigmoid

        elif self.ctype=='multi':
            out_node = 10
            self.last_layer = F.softmax

        elif self.ctype=='regression':
            out_node = 1
            self.last_layer = nn.Identity()

        else:
            print("Choose between 'binary', 'multi', 'regression'")


        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.filters[0], kernel_size=7, stride=2),
            nn.BatchNorm2d(self.filters[0]),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(self.filters[0], self.filters[1], kernel_size=3, stride=1),
            nn.BatchNorm2d(self.filters[1]),
            nn.ReLU(),
            nn.Dropout(.2),

            nn.Conv2d(self.filters[1], self.filters[1], kernel_size=3, stride=1),
            nn.BatchNorm2d(self.filters[1]),
            nn.ReLU(),
            nn.Dropout(.2),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(self.filters[1], self.filters[2], kernel_size=3, stride=1),
            nn.BatchNorm2d(self.filters[2]),
            nn.ReLU(),
            nn.Dropout(.2),

            nn.Conv2d(self.filters[2], self.filters[2], kernel_size=3, stride=1),
            nn.BatchNorm2d(self.filters[2]),
            nn.ReLU(),
            nn.Dropout(.2),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(self.filters[2], self.filters[3], kernel_size=self.kernel_size, stride=self.stride),
            nn.BatchNorm2d(self.filters[3]),
            nn.ReLU(),
            nn.Dropout(.2),

            nn.Conv2d(self.filters[3], self.filters[3], kernel_size=self.kernel_size, stride=self.stride),
            nn.BatchNorm2d(self.filters[3]),
            nn.ReLU(),
            nn.Dropout(.2),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        if flat_nodes:

            self.avgpool = False
            self.flat_nodes = flat_nodes
            self.fc = nn.Sequential(
                nn.Linear(self.flat_nodes, 100),
                nn.Linear(100, out_node)
            )


        else:

            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))

            if kernel_size in [5, 7]:
                self.flat_nodes = 3 * 4 * self.filters[2]

            elif kernel_size == 3:
                self.flat_nodes = 4 * 5 * self.filters[2]

            else:
                self.flat_nodes = self.cal_cnn(F=self.kernel_size)

            self.fc = nn.Sequential(
                nn.Linear(3 * 3 * 64, 100),
                nn.Linear(100, out_node)
            )


    def forward(self, x):

        # Convolution
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.avgpool: out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)

        # Fully-Connected Layer
        out = self.fc(out)
        out = self.last_layer(out)

        return out


    def loss_fn(self, weight=None):

        # TODO: add weight for binary
        if self.ctype == 'binary':
            return nn.BCELoss()

        elif self.ctype == 'multi':
            if weight is not None: return nn.CrossEntropyLoss(weight=weight)
            else: return nn.CrossEntropyLoss()

        elif self.ctype == 'regression':
            return nn.L1Loss()

        else:
            pass


    def accuracy(self, pred, true, confusion=False):

        if self.ctype=='binary':
            correct, cnt = 0, 0
            tp, tn, fp, fn = 0, 0, 0, 0
            for p, t in zip(pred, true):
                cnt += 1
                if (p >= 0.5) & (t == 1):
                    correct += 1
                    tp += 1

                elif (p >= 0.5) & (t == 0):
                    fp += 1

                elif (p < 0.5) & (t == 1):
                    tn += 1

                elif (p < 0.5) & (t == 0):
                    correct += 1
                    fn += 1

                else:
                    pass

            if confusion:
                return correct, cnt, np.array([tp, tn, fp, fn])

            else:
                return correct, cnt

        elif self.ctype=='multi':
            result, cnt = 0, 0
            for p, t in zip(pred.argmax(axis=1), true):
                cnt += 1
                if p == t:
                    result += 1

            return result, cnt

        elif self.ctype=='regression':
            pass

        else:
            pass

    def cal_cnn(self, W=307, H=375, F=7, S=2, P=0):

        W1 = round((W - F + P) / S)
        W1 /= 2
        W2 = round((W1 - F + P) / S)
        W2 /= 2
        W3 = round((W2 - F + P) / S)
        W3 /= 2

        H1 = round((H - F + P) / S)
        H1 /= 2
        H2 = round((H1 - F + P) / S)
        H2 /= 2
        H3 = round((H2 - F + P) / S)
        H3 /= 2

        return round(W3), round(H3)