import pandas as pd
import numpy as np
from io import BytesIO
import random
import torch
import torch.nn as nn
import torch.optim as optim
from io import StringIO

## cu model-a

#model-a eu #만성
class MLPModel(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
    def __init__(self): 
        super(MLPModel, self).__init__()
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

def eu_a(ph,na,ca,mg,al,doc,dc):
    pH = ph
    Na = na
    Ca = ca
    Mg = mg
    Alkalinity = al
    Dissolved_organic_carbon = doc
    Individual_dissolved_Cu =dc
    # Annual_dissolved_Cu =
    # Electronical_conductivity =
    model = MLPModel()
    model.load_state_dict(torch.load('Cu_full-para_00.pth'))

    list=[float(pH),float(Na),float(Mg),float(Ca),float(Dissolved_organic_carbon),float(Alkalinity)]
    data=np.array(list)
    kr_test_input_x = torch.Tensor(data)
    kr_test_reshape_NN = torch.reshape(model(kr_test_input_x), (-1,))
    result1=np.power(10,(np.log10(kr_test_reshape_NN.item())-0.41)/0.85)

    output_cu_eu_a_a= float(result1)
    x_cu_eu_a=float(Individual_dissolved_Cu)/result1

    return output_cu_eu_a_a,x_cu_eu_a

def eu_a_csv(dataframe):

    model = MLPModel()
    model.load_state_dict(torch.load('Cu_full-para_00.pth'))
    kr_test_data = dataframe
    kr_test_data2=kr_test_data.copy()

    if 'Annual dissolved Cu (㎍/L)' in kr_test_data and kr_test_data['Annual dissolved Cu (㎍/L)'].notnull().all() :
        kr_test_data = kr_test_data[["pH", "Na (mg/L)", "Mg (mg/L)", "Ca (mg/L)", "DOC (mg/L)", "Alkalinity (㎎ CaCO3/L)"]] 
        kr_test_X = kr_test_data.iloc[:,:6]

        kr_test_input_x = torch.Tensor(kr_test_X.values)

        kr_test_reshape_NN = torch.reshape(model(kr_test_input_x), (-1,))
        a = np.power(10, (np.log10(kr_test_reshape_NN.detach().numpy()) - 0.41) / 0.85)
        a = pd.DataFrame(a)
        a.columns=['BLM-based chronic PNEC']
        a=pd.merge(kr_test_data,a,left_index=True,right_index=True)
        a['Chronic RCR'] = kr_test_data2['Annual dissolved Cu (㎍/L)'] / a['BLM-based chronic PNEC'] 

        df=a
    else:
        kr_test_data = kr_test_data[["pH", "Na (mg/L)", "Mg (mg/L)", "Ca (mg/L)", "DOC (mg/L)", "Alkalinity (㎎ CaCO3/L)"]] 

        kr_test_X = kr_test_data.iloc[:,:6]

        kr_test_input_x = torch.Tensor(kr_test_X.values)

        kr_test_reshape_NN = torch.reshape(model(kr_test_input_x), (-1,))
        a = np.power(10, (np.log10(kr_test_reshape_NN.detach().numpy()) - 0.41) / 0.85)
        a = pd.DataFrame(a)
        a.columns=['BLM-based chronic PNEC']
        a=pd.merge(kr_test_data,a,left_index=True,right_index=True)

        df=a
    return df

#model-a asian #급성

class MLPModel_A_Cu(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
    def __init__(self): 
        super(MLPModel_A_Cu, self).__init__()
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
    
def asia_a(ph,na,ca,mg,al,doc,dc):
    pH = ph
    Na = na
    Ca = ca
    Mg = mg
    Alkalinity = al
    Dissolved_organic_carbon = doc
    #Individual_dissolved_Cu =dc
    Annual_dissolved_Cu =dc
    # Electronical_conductivity =
    model_a_cu = MLPModel_A_Cu()
    model_a_cu.load_state_dict(torch.load('Cu_full-para_00.pth'))

    list=[float(pH),float(Na),float(Mg),float(Ca),float(Dissolved_organic_carbon),float(Alkalinity)]
    data=np.array(list)
    kr_test_input_x = torch.Tensor(data)
    kr_test_reshape_NN = torch.reshape(model_a_cu(kr_test_input_x), (-1,))
    output_a_cu_asia=kr_test_reshape_NN.item()

    output_a_asia_a= float(output_a_cu_asia)
    x_a_asia=float(Annual_dissolved_Cu)/(output_a_cu_asia)
    return output_a_asia_a,x_a_asia

def asia_a_csv(dataframe):

    model_a_cu = MLPModel_A_Cu()
    model_a_cu.load_state_dict(torch.load('Cu_full-para_00.pth'))
    kr_test_data = dataframe
    kr_test_data2=kr_test_data.copy()

    if 'Individual dissolved Cu (㎍/L)' in kr_test_data and kr_test_data['Individual dissolved Cu (㎍/L)'].notnull().all() :
        kr_test_data = kr_test_data[["pH", "Na (mg/L)", "Mg (mg/L)", "Ca (mg/L)", "DOC (mg/L)", "Alkalinity (㎎ CaCO3/L)"]] 
        kr_test_X = kr_test_data.iloc[:,:6]

        kr_test_input_x = torch.Tensor(kr_test_X.values)

        kr_test_reshape_NN = torch.reshape(model_a_cu(kr_test_input_x), (-1,))
        a = kr_test_reshape_NN.detach().numpy()
        a = pd.DataFrame(a)
        a.columns=['BLM-based acute PNEC']
        a=a[['BLM-based acute PNEC']]
        a=pd.merge(kr_test_data,a,left_index=True,right_index=True)
        a['Acute RCR']= kr_test_data2['Individual dissolved Cu (㎍/L)'] / a['BLM-based acute PNEC'] 
        df=a[['BLM-based acute PNEC','Acute RCR']]
    else:
        kr_test_data = kr_test_data[["pH", "Na (mg/L)", "Mg (mg/L)", "Ca (mg/L)", "DOC (mg/L)", "Alkalinity (㎎ CaCO3/L)"]] 

        kr_test_X = kr_test_data.iloc[:,:6]

        kr_test_input_x = torch.Tensor(kr_test_X.values)

        kr_test_reshape_NN = torch.reshape(model_a_cu(kr_test_input_x), (-1,))
        a = kr_test_reshape_NN.detach().numpy()
        a = pd.DataFrame(a)
        a.columns=['BLM-based acute PNEC']
        a=a[['BLM-based acute PNEC']]
        a=pd.merge(kr_test_data,a,left_index=True,right_index=True)

        df=a[['BLM-based acute PNEC']]
    return df

# ## cu model-b #만성 -eu

class MLPModel_B(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
    def __init__(self): 
        super(MLPModel_B, self).__init__()
        self.linear1 = nn.Linear(5,20)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(20,15)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(15,10)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(10,1)

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

def eu_b(ph,na,ca,mg,doc,dc):
    pH = ph
    Na = na
    Ca = ca
    Mg = mg
    # Alkalinity = al
    Dissolved_organic_carbon = doc
    Individual_dissolved_Cu =dc
    # Annual_dissolved_Cu =
    # Electronical_conductivity =
    model_B = MLPModel_B()
    model_B.load_state_dict(torch.load('no_alk_training_model_kor.pth'))     
    
    list=[float(pH),float(Na),float(Mg),float(Ca),float(Dissolved_organic_carbon)]
    data=np.array(list)
    kr_test_input_x = torch.Tensor(data)
    kr_test_reshape_NN = torch.reshape(model_B(kr_test_input_x), (-1,))
    result_b_eu=np.power(10,(np.log10(kr_test_reshape_NN.item())-0.41)/0.85)

    
    output_cu_eu_b_b=float(result_b_eu)
    x_cu_eu_b=float(Individual_dissolved_Cu)/result_b_eu
    
    return output_cu_eu_b_b,x_cu_eu_b

def eu_b_csv(dataframe):

    model_B = MLPModel_B()
    model_B.load_state_dict(torch.load('no_alk_training_model_kor.pth'))     
    kr_test_data = dataframe
    kr_test_data2=kr_test_data.copy()

    if 'Annual dissolved Cu (㎍/L)' in kr_test_data and kr_test_data['Annual dissolved Cu (㎍/L)'].notnull().all() :
        kr_test_data = kr_test_data[["pH", "Na (mg/L)", "Mg (mg/L)", "Ca (mg/L)", "DOC (mg/L)"]] 

        kr_test_X = kr_test_data.iloc[:,:5]

        kr_test_input_x = torch.Tensor(kr_test_X.values)

        kr_test_reshape_NN = torch.reshape(model_B(kr_test_input_x), (-1,))
        a = np.power(10, (np.log10(kr_test_reshape_NN.detach().numpy()) - 0.41) / 0.85)
        a = pd.DataFrame(a)
        a.columns=['BLM-based chronic PNEC']
        a=pd.merge(kr_test_data,a,left_index=True,right_index=True)
        a['Chronic RCR']= kr_test_data2['Annual dissolved Cu (㎍/L)'] / a['BLM-based chronic PNEC'] 

        df=a
    else:
        kr_test_data = kr_test_data[["pH", "Na (mg/L)", "Mg (mg/L)", "Ca (mg/L)", "DOC (mg/L)"]] 

        kr_test_X = kr_test_data.iloc[:,:5]

        kr_test_input_x = torch.Tensor(kr_test_X.values)

        kr_test_reshape_NN = torch.reshape(model_B(kr_test_input_x), (-1,))
        a = np.power(10, (np.log10(kr_test_reshape_NN.detach().numpy()) - 0.41) / 0.85)
        a = pd.DataFrame(a)
        a.columns=['BLM-based chronic PNEC']
        a=pd.merge(kr_test_data,a,left_index=True,right_index=True)

        df=a
    return df

def asia_b(ph,na,ca,mg,doc,dc):
    pH = ph
    Na = na
    Ca = ca
    Mg = mg
    Dissolved_organic_carbon = doc
    Individual_dissolved_Cu =dc
    model_B = MLPModel_B()
    model_B.load_state_dict(torch.load('no_alk_training_model_kor.pth'))     
    
    list=[float(pH),float(Na),float(Mg),float(Ca),float(Dissolved_organic_carbon)]
    data=np.array(list)
    kr_test_input_x = torch.Tensor(data)
    kr_test_reshape_NN = torch.reshape(model_B(kr_test_input_x), (-1,))
    result_b_asia=kr_test_reshape_NN.item()

    
    output_cu_asia_b_b=float(result_b_asia)
    x_cu_asia_b=float(Individual_dissolved_Cu)/(result_b_asia)
    
    return output_cu_asia_b_b,x_cu_asia_b

### asia csv
def asia_b_csv(dataframe):

    model_B = MLPModel_B()
    model_B.load_state_dict(torch.load('no_alk_training_model_kor.pth'))     
    kr_test_data = dataframe
    kr_test_data2=kr_test_data.copy()

    if 'Individual dissolved Cu (㎍/L)' in kr_test_data and kr_test_data['Individual dissolved Cu (㎍/L)'].notnull().all() :
        kr_test_data = kr_test_data[["pH", "Na (mg/L)", "Mg (mg/L)", "Ca (mg/L)", "DOC (mg/L)"]] 

        kr_test_X = kr_test_data.iloc[:,:5]

        kr_test_input_x = torch.Tensor(kr_test_X.values)

        kr_test_reshape_NN = torch.reshape(model_B(kr_test_input_x), (-1,))
        a = kr_test_reshape_NN.detach().numpy()
        a = pd.DataFrame(a)
        a.columns=['BLM-based acute PNEC']
        a=a[['BLM-based acute PNEC']]
        a=pd.merge(kr_test_data,a,left_index=True,right_index=True)
        a['Acute RCR']=kr_test_data2['Individual dissolved Cu (㎍/L)'] / a['BLM-based acute PNEC'] 

        df=a[['BLM-based acute PNEC','Acute RCR']]
    else:
        kr_test_data = kr_test_data[["pH", "Na (mg/L)", "Mg (mg/L)", "Ca (mg/L)", "DOC (mg/L)"]] 

        kr_test_X = kr_test_data.iloc[:,:5]

        kr_test_input_x = torch.Tensor(kr_test_X.values)

        kr_test_reshape_NN = torch.reshape(model_B(kr_test_input_x), (-1,))
        a = kr_test_reshape_NN.detach().numpy()
        a = pd.DataFrame(a)
        a.columns=['BLM-based acute PNEC']
        a=a[['BLM-based acute PNEC']]
        a=pd.merge(kr_test_data,a,left_index=True,right_index=True)

        df=a[['BLM-based acute PNEC']]
    return df


# ## cu model-c

# #model-c-eu #만성
class MLPModel_C(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
    def __init__(self): 
        super(MLPModel_C, self).__init__()
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

def eu_c(ph,ec,doc,dc):

    pH=ph
    Electronical_conductivity=ec    
    Dissolved_organic_carbon=doc
    Individual_dissolved_Cu=dc

    model_C = MLPModel_C()
    model_C.load_state_dict(torch.load('cond_Cu_Kor00.pth'))

    x=float(Electronical_conductivity)
    y=float(Dissolved_organic_carbon)
    list=[float(pH),np.log10(x),np.log10(y)]
    data=np.array(list)
    kr_test_input_x = torch.Tensor(data)
    kr_test_reshape_NN = torch.reshape(model_C(kr_test_input_x), (-1,))

    output_c_eu_cu=float(np.power(10,(np.log10(kr_test_reshape_NN.item())-0.41)/0.85))

    x_cu_eu_c=float(Individual_dissolved_Cu)/output_c_eu_cu


    return output_c_eu_cu,x_cu_eu_c


#eu_c csv
def eu_c_csv(dataframe):


    model_C = MLPModel_C()
    model_C.load_state_dict(torch.load('cond_Cu_Kor00.pth'))

    kr_test_data = dataframe
    kr_test_data3=kr_test_data.copy()

    if 'Annual dissolved Cu (㎍/L)' in kr_test_data and kr_test_data['Annual dissolved Cu (㎍/L)'].notnull().all() :
        kr_test_data = kr_test_data[["pH", "Cond (㎲/㎝)", "DOC (mg/L)",]] 
        kr_test_data['Cond (㎲/㎝)'] = np.log10(kr_test_data['Cond (㎲/㎝)'])
        kr_test_data['DOC (mg/L)'] = np.log10(kr_test_data['DOC (mg/L)'])
        kr_test_X = kr_test_data.iloc[:,:3]

        kr_test_input_x = torch.Tensor(kr_test_X.values)

        kr_test_reshape_NN = torch.reshape(model_C(kr_test_input_x), (-1,))

        a = np.power(10, (np.log10(kr_test_reshape_NN.detach().numpy()) - 0.41) / 0.85)
        a = pd.DataFrame(a)
        a.columns=['BLM-based chronic PNEC']
        a=pd.merge(kr_test_data3,a,left_index=True,right_index=True)
        a['Chronic RCR']= kr_test_data3['Annual dissolved Cu (㎍/L)'] / a['BLM-based chronic PNEC'] 

        df=a

    else:
        kr_test_data = kr_test_data[["pH", "Cond (㎲/㎝)", "DOC (mg/L)",]] 
        kr_test_data['Cond (㎲/㎝)'] = np.log10(kr_test_data['Cond (㎲/㎝)'])
        kr_test_data['DOC (mg/L)'] = np.log10(kr_test_data['DOC (mg/L)'])
        kr_test_X = kr_test_data.iloc[:,:3]

        kr_test_input_x = torch.Tensor(kr_test_X.values)

        kr_test_reshape_NN = torch.reshape(model_C(kr_test_input_x), (-1,))

        a = np.power(10, (np.log10(kr_test_reshape_NN.detach().numpy()) - 0.41) / 0.85)
        a = pd.DataFrame(a)
        a.columns=['BLM-based chronic PNEC']
        a=pd.merge(kr_test_data3,a,left_index=True,right_index=True)

        df=a
    return df

# #model-c-asia #급성
class MLPModel_C_Cu(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
    def __init__(self): 
        super(MLPModel_C_Cu, self).__init__()
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

def asia_c(ph,ec,doc,dc):

    pH=ph
    Electronical_conductivity=ec    
    Dissolved_organic_carbon=doc
    Individual_dissolved_Cu=dc

    model_C_Cu = MLPModel_C_Cu()
    model_C_Cu.load_state_dict(torch.load('cond_Cu_Kor00.pth'))

    x=float(Electronical_conductivity)
    y=float(Dissolved_organic_carbon)
    list=[float(pH),np.log10(x),np.log10(y)]
    data=np.array(list)
    kr_test_input_x = torch.Tensor(data)
    kr_test_reshape_NN = torch.reshape(model_C_Cu(kr_test_input_x), (-1,))

    output_c_asia_cu=float(kr_test_reshape_NN.item())

    x_cu_asia_c=float(Individual_dissolved_Cu)/output_c_asia_cu


    return output_c_asia_cu,x_cu_asia_c

### asia_c csv
def asia_c_csv(dataframe):


    model_C_Cu = MLPModel_C_Cu()
    model_C_Cu.load_state_dict(torch.load('cond_Cu_Kor00.pth'))

    kr_test_data = dataframe
    kr_test_data3=kr_test_data.copy()

    if 'Individual dissolved Cu (㎍/L)' in kr_test_data and kr_test_data['Individual dissolved Cu (㎍/L)'].notnull().all():
        kr_test_data = kr_test_data[["pH", "Cond (㎲/㎝)", "DOC (mg/L)",]] 
        kr_test_data['Cond (㎲/㎝)'] = np.log10(kr_test_data['Cond (㎲/㎝)'])
        kr_test_data['DOC (mg/L)'] = np.log10(kr_test_data['DOC (mg/L)'])
        kr_test_X = kr_test_data.iloc[:,:3]

        kr_test_input_x = torch.Tensor(kr_test_X.values)

        kr_test_reshape_NN = torch.reshape(model_C_Cu(kr_test_input_x), (-1,))

        a = kr_test_reshape_NN.detach().numpy()
        a = pd.DataFrame(a)
        a.columns=['BLM-based acute PNEC']
        a=a[['BLM-based acute PNEC']]
        a=pd.merge(kr_test_data3,a,left_index=True,right_index=True)
        a['Acute RCR']= kr_test_data3['Individual dissolved Cu (㎍/L)'] / a['BLM-based acute PNEC']  

        df=a[['BLM-based acute PNEC','Acute RCR']]

    else:
        kr_test_data = kr_test_data[["pH", "Cond (㎲/㎝)", "DOC (mg/L)",]] 
        kr_test_data['Cond (㎲/㎝)'] = np.log10(kr_test_data['Cond (㎲/㎝)'])
        kr_test_data['DOC (mg/L)'] = np.log10(kr_test_data['DOC (mg/L)'])
        kr_test_X = kr_test_data.iloc[:,:3]

        kr_test_input_x = torch.Tensor(kr_test_X.values)

        kr_test_reshape_NN = torch.reshape(model_C_Cu(kr_test_input_x), (-1,))

        a = kr_test_reshape_NN.detach().numpy()
        a = pd.DataFrame(a)
        a.columns=['BLM-based acute PNEC']
        a=a[['BLM-based acute PNEC']]
        a=pd.merge(kr_test_data3,a,left_index=True,right_index=True)

        df=a[['BLM-based acute PNEC']]
    return df