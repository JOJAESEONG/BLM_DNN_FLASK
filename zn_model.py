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

## mdoel-zn-asia-a #급성

class MLPModel_A_Zn(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
    def __init__(self): 
        super(MLPModel_A_Zn, self).__init__()
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
    
model_A_Zn = MLPModel_A_Zn()
model_A_Zn.load_state_dict(torch.load('Zn_full-para_00.pth'))

list=[float(pH),float(Na),float(Mg),float(Ca),float(Dissolved_organic_carbon),float(Alkalinity)]
data=np.array(list)
kr_test_input_x = torch.Tensor(data)
kr_test_reshape_NN = torch.reshape(model_A_Zn(kr_test_input_x), (-1,))

output_zn_asia_a =kr_test_reshape_NN.item()

output_zn_asia_a_a=float(Annual_dissolved_Cu)/output_zn_asia_a 

## mdoel-zn-asia-c #급성

class MLPModel_C_Zn(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
    def __init__(self): 
        super(MLPModel_C_Zn, self).__init__()
        self.linear1 = nn.Linear(3,12)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(12,9)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(9,6)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(6,1)

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
       
model_C_Zn = MLPModel_C_Zn()
model_C_Zn.load_state_dict(torch.load('cond_Zn_Kor00.pth'))

x=float(Electronical_conductivity)
y=float(Dissolved_organic_carbon)
list=[float(pH),np.log10(x),np.log10(y)]
data=np.array(list)
kr_test_input_x = torch.Tensor(data)
kr_test_reshape_NN = torch.reshape(model_C_Zn(kr_test_input_x), (-1,))

output_zn_asia_c=kr_test_reshape_NN.item()

output_zn_asia_c_c=float(Annual_dissolved_Cu)/output_zn_asia_c 