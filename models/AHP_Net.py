import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class AHP_HP_RES50(nn.Module):

    def __init__(self):
        super(AHP_HP_RES50, self).__init__()

        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.bn3 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, input):

        x = self.bn1(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x



class AHP_HP_RES50_PRESIGMOID(nn.Module):

    def __init__(self):
        super(AHP_HP_RES50_PRESIGMOID, self).__init__()

        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.bn3 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, input):

        x = self.bn1(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x_pre = self.fc3(x)
        x = F.sigmoid(x_pre)
        return x, x_pre



class AHP_HP_VGG16(nn.Module):

    def __init__(self):
        super(AHP_HP_VGG16, self).__init__()

        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.bn3 = nn.BatchNorm1d(100)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, input):

        x = self.bn1(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x




# auxiliary hardness prediction network hardness predictor
def ahp_net_hp_res50():
    model = AHP_HP_RES50()
    return model

def ahp_net_hp_res50d():
    model = AHP_HP_RES50D()
    return model

def ahp_net_hp_res50b():
    model = AHP_HP_RES50B()
    return model


def ahp_net_hp_res50_added():
    model = AHP_HP_RES50_ADDED()
    return model


def ahp_net_hp_res50_presigmoid():
    model = AHP_HP_RES50_PRESIGMOID()
    return model

def ahp_net_hp_res50_share():
    model = AHP_HP_RES50_SHARE()
    return model


# auxiliary hardness prediction network hardness sample filter
def ahp_net_hsf_res50():
    model = AHP_HSF_RES50()
    return model

def ahp_net_hp_vgg16():
    model = AHP_HP_VGG16()
    return model