import numpy as np

from torch import nn, optim
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self, ctype, kernel_size=5, stride=2, save=False):
        super(ConvNet, self).__init__()

        self.ctype = ctype
        self.kernel_size = kernel_size
        self.stride = stride
        self.save = save

        if self.ctype=='binary':
            out_node = 1
            self.last_layer = F.sigmoid

        elif self.ctype=='multi':
            out_node = 10
            self.last_layer = F.sigmoid

        elif self.ctype=='regression':
            out_node = 1
            self.last_layer = nn.Identity()

        else:
            print("Choose between 'binary', 'multi', 'regression'")


        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 10, kernel_size=self.kernel_size, stride=self.stride),
            nn.ReLU(),
            nn.BatchNorm2d(10), # takes number of kernel as input
            nn.Dropout(.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=self.kernel_size, stride=self.stride),
            nn.ReLU(),
            nn.BatchNorm2d(20), # takes number of kernel as input
            nn.Dropout(.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=self.kernel_size, stride=self.stride),
            nn.ReLU(),
            nn.BatchNorm2d(40), # takes number of kernel as input
            nn.Dropout(.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(480, 100)
        self.fc2 = nn.Linear(100, out_node)

    def forward(self, x):

        # Convolution
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)

        # Fully-Connected Layer
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.last_layer(out)

        return out


    def accuracy(self, pred, true):

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

            return correct, cnt, np.array([tp, tn, fp, fn])

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