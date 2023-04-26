import pandas as pd
import numpy as np
from io import BytesIO
import random
import torch
import torch.nn as nn
import torch.optim as optim
from io import StringIO

## 입력자료

pH =
Na =
Ca =
Mg =
Alkalinity =
Dissolved_organic_carbon =
Individual_dissolved_Cu =
Annual_dissolved_Cu =
Electronical_conductivity =


#### Ni_model-asia-A #급성
class MLPModel_A_Ni(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
    def __init__(self): 
        super(MLPModel_A_Ni, self).__init__()
        self.linear1 = nn.Linear(6,24)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(24,18)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(18,12)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(12,1)

    def forward(self, x):
    # 인스턴스(샘플) x가 인풋으로 들어왔을 때 모델이 예측하는 y값을 리턴합니다.
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

model_A_Ni = MLPModel_A_Ni()
model_A_Ni.load_state_dict(torch.load('Ni_full-para_00.pth'))

list=[float(pH),float(Na),float(Mg),float(Ca),float(Dissolved_organic_carbon),float(Alkalinity)]
data=np.array(list)
kr_test_input_x = torch.Tensor(data)
kr_test_reshape_NN = torch.reshape(model_A_Ni(kr_test_input_x), (-1,))
output_Ni_a=kr_test_reshape_NN.item()

x_ni_asia_a=float(Annual_dissolved_Cu)/output_Ni_a

#### Ni_model-asia-c #급성

class MLPModel_C_Ni(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
    def __init__(self):
        super(MLPModel_C_Ni, self).__init__()
        self.linear1 = nn.Linear(4,32)
        self.sigmoid1 = nn.ReLU()
        self.linear2 = nn.Linear(32,24)
        self.sigmoid2 = nn.ReLU()
        self.linear3 = nn.Linear(24,16)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(16,1)

    def forward(self, x):
    # 인스턴스(샘플) x가 인풋으로 들어왔을 때 모델이 예측하는 y값을 리턴합니다.
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x
    
model_C_Ni = MLPModel_C_Ni()
model_C_Ni.load_state_dict(torch.load('cond_Ni_Kor(log)02.pth'))

x=float(Electronical_conductivity)
y=float(Dissolved_organic_carbon)
P_Ak= -1.954980 +float(pH) * 0.161542 + np.log10(y) * 0.119819 + np.log10(x)  * 0.988312
list=[float(pH),np.log10(x),np.log10(y),P_Ak]
data=np.array(list)
kr_test_input_x = torch.Tensor(data)
kr_test_reshape_NN = torch.reshape(model_C_Ni(kr_test_input_x), (-1,))

output_Ni_c=kr_test_reshape_NN.item()

x_ni_asia_c=float(Annual_dissolved_Cu)/output_Ni_c

