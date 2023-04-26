import pandas as pd
import panel as pn
import numpy as np
from io import BytesIO
import random
import torch
import torch.nn as nn
import torch.optim as optim
from io import StringIO

css = ['https://cdn.datatables.net/1.10.24/css/jquery.dataTables.min.css',
       # Below: Needed for export buttons
       'https://cdn.datatables.net/buttons/1.7.0/css/buttons.dataTables.min.css'
]

css2 = '''
.bk.panel-widget {
  border: None;
  font-size: 20px;
}

.button .bk-btn{
  font-size:20px;
  font-family: NanumBarunGothic;
}

.widget-button .bk-btn {
  font-size:20px;
  font-family: NanumBarunGothic;
}

.table .tabulator {
  font-size: 20px;
}

'''



js = {
    '$': 'https://code.jquery.com/jquery-3.5.1.js',
    'DataTable': 'https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js',
    # Below: Needed for export buttons
    'buttons': 'https://cdn.datatables.net/buttons/1.7.0/js/dataTables.buttons.min.js',
    'jszip': 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.1.3/jszip.min.js',
    'pdfmake': 'https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.53/pdfmake.min.js',
    'vfsfonts': 'https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.53/vfs_fonts.js',
    'html5buttons': 'https://cdn.datatables.net/buttons/1.7.0/js/buttons.html5.min.js',
}
pn.extension(sizing_mode='stretch_width',css_files=css, raw_css=[css2], js_files=js)

optionss=['MAIN', 'European species', 'Asian species' ]

select_main=pn.widgets.Select(name="Main_Feature_Selector", options=optionss, value='MAIN', sizing_mode='fixed',width=300, css_classes=['panel-widget'])
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
    
model = MLPModel()
model.load_state_dict(torch.load('full-para_training_model.pth'))
text_input = pn.widgets.TextInput(name='pH', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input2 = pn.widgets.TextInput(name='Ca (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input3 = pn.widgets.TextInput(name='Mg (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input4 = pn.widgets.TextInput(name='Na (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input5 = pn.widgets.TextInput(name='Alkalinity (㎎ CaCo3/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input6 = pn.widgets.TextInput(name='Dissolved organic carbon (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input7 = pn.widgets.TextInput(name='Dissolved Cu (㎍/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
widget_box=pn.Column(pn.pane.Markdown('### Input variables'),pn.Column(pn.Row(text_input, text_input2, text_input3),pn.Row(text_input4, text_input5, text_input6, text_input7)))

def calculate_A():
    if text_input.value and text_input2.value and text_input3.value and text_input4.value and text_input5.value and text_input6.value =='': 
        kr_test_reshape_NN = 'NONE'
    else :
        list=[float(text_input.value),float(text_input4.value),float(text_input3.value),float(text_input2.value),float(text_input6.value),float(text_input5.value)]
        data=np.array(list)
        kr_test_input_x = torch.Tensor(data)
        kr_test_reshape_NN = torch.reshape(model(kr_test_input_x), (-1,))
    return kr_test_reshape_NN.item()

output1 = pn.widgets.TextInput(name='Copper BLM-based PNECs',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])
output2 = pn.widgets.TextInput(name='RCR',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button2 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def button_event(event):
    output1.value= str(calculate_A())
button.on_click(button_event)

def button_event2(event):
    x=float(text_input7.value)/calculate_A()
    output2.value= str(x)
button2.on_click(button_event2)

main = pn.Column(widget_box,pn.Column(pn.pane.Markdown('<br>'),pn.Row(pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output1,button)),pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output2,button2)))))
file_input = pn.widgets.FileInput(accept='.csv,.json',sizing_mode='fixed', css_classes=['panel-widget'])
button7 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def csv_input():
    if file_input.value is None:
        stock_file = 'kr_test.csv'
    else : 
        stock_file = BytesIO()
        stock_file.write(file_input.value)
        stock_file.seek(0)
    return stock_file

@pn.depends(button7.param.clicks)
def calculate_A_batch(_):
    if csv_input() == 'kr_test.csv':
        kr_test_data = pd.read_csv(csv_input())
        kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC", "Alkalinity"]] 

        df=kr_test_data

        script = """
        <script>
        if (document.readyState === "complete") {
        $('.example').DataTable();
        } else {
        $(document).ready(function () {
            $('.example').DataTable();
        })
        }
        document.oncontextmenu=function(){return false;}
        document.onselectstart=function(){return false;}
        document.ondragstart=function(){return false;}
        </script>
        """
        html = df.to_html(classes=['example', 'panel-df'])
        test=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

    else:
        kr_test_data = pd.read_csv(csv_input())
        kr_test_data2=kr_test_data.copy()
        if 'Dissolved Cu' in kr_test_data :
            kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC", "Alkalinity"]] 
            kr_test_X = kr_test_data.iloc[:,:6]

            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model(kr_test_input_x), (-1,))
            a = kr_test_reshape_NN.detach().numpy()
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data,a,left_index=True,right_index=True)
            a['RCR']=a['BLM-based PNECs'] / kr_test_data2['Dissolved Cu']

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))
            
        else:
            kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC", "Alkalinity"]] 

            kr_test_X = kr_test_data.iloc[:,:6]

            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model(kr_test_input_x), (-1,))
            a = kr_test_reshape_NN.detach().numpy()
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data,a,left_index=True,right_index=True)

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}        
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

        # def file():
        #     sio=StringIO()
        #     df.to_csv(sio, encoding="EUC-KR")
        #     sio.seek(0)
        #     return sio

        # file_download=pn.widgets.FileDownload(callback=file, filename="Calculate_MODEL_data.csv",sizing_mode='fixed')
        test=pn.Column(table)

    return test
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
    
    
model_B = MLPModel_B()
model_B.load_state_dict(torch.load('no_alk_training_model_v1.pth'))
#model(train_input_x)
text_input8 = pn.widgets.TextInput(name='pH', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input9 = pn.widgets.TextInput(name='Ca (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input10 = pn.widgets.TextInput(name='Mg (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input11 = pn.widgets.TextInput(name='Na (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input12 = pn.widgets.TextInput(name='Dissolved organic carbon (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input13 = pn.widgets.TextInput(name='Dissolved Cu (㎍/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
widget_box2=pn.Column(pn.pane.Markdown('### Input variables'),pn.Column(pn.Row(text_input8, text_input9, text_input10),pn.Row( text_input11, text_input12, text_input13)))
def calculate_B():
    if text_input8.value and text_input9.value and text_input10.value and text_input11.value and text_input12.value =='': 
        kr_test_reshape_NN = 0
    else :
        list=[float(text_input8.value),float(text_input11.value),float(text_input10.value),float(text_input9.value),float(text_input12.value)]
        data=np.array(list)
        kr_test_input_x = torch.Tensor(data)
        kr_test_reshape_NN = torch.reshape(model_B(kr_test_input_x), (-1,))
    return kr_test_reshape_NN.item()
output3 = pn.widgets.TextInput(name='Copper BLM-based PNECs',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button3 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])
output4 = pn.widgets.TextInput(name='RCR',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button4 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])
def button_event3(event):
    output3.value= str(calculate_B())
button3.on_click(button_event3)
def button_event4(event):
    x=float(text_input13.value)/calculate_B()
    output4.value= str(x)
button4.on_click(button_event4)
main2 = pn.Column(widget_box2,pn.Column(pn.pane.Markdown('<br>'),pn.Row(pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output3,button3)),pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output4,button4)))))
file_input_B = pn.widgets.FileInput(accept='.csv,.json',sizing_mode='fixed', css_classes=['panel-widget'])
button8 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def csv_input_B():
    if file_input_B.value is None:
        stock_file = 'kr_test.csv'
    else : 
        stock_file = BytesIO()
        stock_file.write(file_input_B.value)
        stock_file.seek(0)
    return stock_file

@pn.depends(button8.param.clicks)
def calculate_B_batch(_):
    if csv_input_B() == 'kr_test.csv':
        kr_test_data = pd.read_csv(csv_input_B())
        kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC"]] 

        df=kr_test_data

        script = """
        <script>
        if (document.readyState === "complete") {
        $('.example').DataTable();
        } else {
        $(document).ready(function () {
            $('.example').DataTable();
        })
        }
        document.oncontextmenu=function(){return false;}
        document.onselectstart=function(){return false;}
        document.ondragstart=function(){return false;}
        </script>
        """
        html = df.to_html(classes=['example', 'panel-df'])
        test=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

    else:
        kr_test_data = pd.read_csv(csv_input_B())
        kr_test_data2=kr_test_data.copy()
        if 'Dissolved Cu' in kr_test_data :
            kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC"]] 

            kr_test_X = kr_test_data.iloc[:,:5]

            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_B(kr_test_input_x), (-1,))
            a = kr_test_reshape_NN.detach().numpy()
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data,a,left_index=True,right_index=True)
            a['RCR']=a['BLM-based PNECs'] / kr_test_data2['Dissolved Cu']

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))
        else:
            kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC"]] 

            kr_test_X = kr_test_data.iloc[:,:5]

            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_B(kr_test_input_x), (-1,))
            a = kr_test_reshape_NN.detach().numpy()
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data,a,left_index=True,right_index=True)

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))
            
        # def file():
        #     sio=StringIO()
        #     df.to_csv(sio, encoding="EUC-KR")
        #     sio.seek(0)
        #     return sio

        # file_download=pn.widgets.FileDownload(callback=file, filename="Calculate_MODEL2_data.csv",sizing_mode='fixed')
        test=pn.Column(table)

    return test
    
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
    
model_C = MLPModel_C()
model_C.load_state_dict(torch.load('cond_training_model_correct_MLR_v1.pth'))
#model(train_input_x)
text_input14 = pn.widgets.TextInput(name='pH', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input15 = pn.widgets.TextInput(name='Electronical conductivity (㎲/㎝)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input16 = pn.widgets.TextInput(name='Dissolved organic carbon (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input17 = pn.widgets.TextInput(name='Dissolved Cu', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
widget_box3=pn.Column(pn.pane.Markdown('### Input variables'),pn.Row(text_input14, text_input15, text_input16, text_input17))
def calculate_C():
    if text_input14.value and text_input15.value and text_input16.value =='': 
        kr_test_reshape_NN = 0
    else :
        x=float(text_input15.value)
        y=float(text_input16.value)
        list=[float(text_input14.value),np.log10(x),np.log10(y)]
        data=np.array(list)
        kr_test_input_x = torch.Tensor(data)
        kr_test_reshape_NN = torch.reshape(model_C(kr_test_input_x), (-1,))
    return kr_test_reshape_NN.item()
output5 = pn.widgets.TextInput(name='Copper BLM-based PNECs',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button5 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])
output6 = pn.widgets.TextInput(name='RCR',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button6 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])
def button_event5(event):
    output5.value= str(calculate_C())
button5.on_click(button_event5)
def button_event6(event):
    x=float(text_input17.value)/calculate_C()
    output6.value= str(x)
button6.on_click(button_event6)
main3 = pn.Column(widget_box3,pn.Column(pn.pane.Markdown('<br>'),pn.Row(pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output5,button5)),pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output6,button6)))))
file_input_C = pn.widgets.FileInput(accept='.csv,.json',sizing_mode='fixed', css_classes=['panel-widget'])
button9 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def csv_input_C():
    if file_input_C.value is None:
        stock_file = 'kr_test.csv'
    else : 
        stock_file = BytesIO()
        stock_file.write(file_input_C.value)
        stock_file.seek(0)
    return stock_file

@pn.depends(button9.param.clicks)
def calculate_C_batch(_):
    if csv_input_C() == 'kr_test.csv':
        kr_test_data = pd.read_csv(csv_input_C())
        kr_test_data = kr_test_data[["pH", "Cond", "DOC",]] 

        df=kr_test_data

        script = """
        <script>
        if (document.readyState === "complete") {
        $('.example').DataTable();
        } else {
        $(document).ready(function () {
            $('.example').DataTable();
        })
        }
        document.oncontextmenu=function(){return false;}
        document.onselectstart=function(){return false;}
        document.ondragstart=function(){return false;}
        </script>
        """
        html = df.to_html(classes=['example', 'panel-df'])
        test=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

    else:
        kr_test_data = pd.read_csv(csv_input_C())
        kr_test_data2=kr_test_data.copy()

        if 'Dissolved Cu' in kr_test_data :
            kr_test_data = kr_test_data[["pH", "Cond", "DOC",]] 
            kr_test_data3=kr_test_data.copy()
            kr_test_data['Cond'] = np.log10(kr_test_data['Cond'])
            kr_test_data['DOC'] = np.log10(kr_test_data['DOC'])
            kr_test_X = kr_test_data.iloc[:,:3]

            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_C(kr_test_input_x), (-1,))

            a = kr_test_reshape_NN.detach().numpy()
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data3,a,left_index=True,right_index=True)
            a['RCR']=a['BLM-based PNECs'] / kr_test_data2['Dissolved Cu']

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

        else:
            kr_test_data = kr_test_data[["pH", "Cond", "DOC",]] 
            kr_test_data3=kr_test_data.copy()
            kr_test_data['Cond'] = np.log10(kr_test_data['Cond'])
            kr_test_data['DOC'] = np.log10(kr_test_data['DOC'])
            kr_test_X = kr_test_data.iloc[:,:3]

            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_C(kr_test_input_x), (-1,))

            a = kr_test_reshape_NN.detach().numpy()
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data3,a,left_index=True,right_index=True)

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

        # def file():
        #     sio=StringIO()
        #     df.to_csv(sio, encoding="EUC-KR")
        #     sio.seek(0)
        #     return sio

        # file_download=pn.widgets.FileDownload(callback=file, filename="Calculate_MODEL3_data.csv",sizing_mode='fixed')
        test=pn.Column(table)

    return test
mark=pn.pane.Markdown("#### DOC: dissolved organic carbon ㎎/L, EC: electronical conductivity ㎲/㎝, Unit of Ca, Mg and Na: dissolved concentration ㎎/L, Unit of alkalinity: ㎎ CaCO3/L")
mark2=pn.pane.Markdown("---")
mark3=pn.pane.Markdown("## Batchfile <br> <br> The CSV file format to be input must be the same as the sample format below <br> * If 'Dissolved Cu' exists in your csv column, The RCR is calculated")
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
    
model_a_cu = MLPModel_A_Cu()
model_a_cu.load_state_dict(torch.load('Cu_full-para_00.pth'))
#model(train_input_x)
text_input18 = pn.widgets.TextInput(name='pH', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input19 = pn.widgets.TextInput(name='Ca (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input20 = pn.widgets.TextInput(name='Mg (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input21 = pn.widgets.TextInput(name='Na (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input22 = pn.widgets.TextInput(name='Alkalinity (㎎ CaCo3/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input23 = pn.widgets.TextInput(name='Dissolved organic carbon (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input24 = pn.widgets.TextInput(name='Dissolved Cu (㎍/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
widget_box4=pn.Column(pn.pane.Markdown('### Input variables'),pn.Column(pn.Row(text_input18, text_input19, text_input20),pn.Row(text_input21, text_input22, text_input23, text_input24)))
def calculate_A_Cu():
    if text_input18.value and text_input19.value and text_input20.value and text_input21.value and text_input22.value and text_input23.value =='': 
        kr_test_reshape_NN = 'NONE'
    else :
        list=[float(text_input18.value),float(text_input21.value),float(text_input20.value),float(text_input19.value),float(text_input23.value),float(text_input22.value)]
        data=np.array(list)
        kr_test_input_x = torch.Tensor(data)
        kr_test_reshape_NN = torch.reshape(model_a_cu(kr_test_input_x), (-1,))
    return np.power(10,(np.log10(kr_test_reshape_NN.item())-0.41)/0.85)
output7 = pn.widgets.TextInput(name='Copper BLM-based PNECs',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button10 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])
output8 = pn.widgets.TextInput(name='RCR',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button11 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])
def button_event_Cu(event):
    output7.value= str(calculate_A_Cu())
button10.on_click(button_event_Cu)
def button_event2_Cu(event):
    x=float(text_input24.value)/calculate_A_Cu()
    output8.value= str(x)
button11.on_click(button_event2_Cu)
main_Cu_A = pn.Column(widget_box4,pn.Column(pn.pane.Markdown('<br>'),pn.Row(pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output7,button10)),pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output8,button11)))))
file_input4 = pn.widgets.FileInput(accept='.csv,.json',sizing_mode='fixed', css_classes=['panel-widget'])
button7_Cu_A = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def csv_input_Cu_A():
    if file_input4.value is None:
        stock_file = 'kr_test(log)_c_test.csv'
    else : 
        stock_file = BytesIO()
        stock_file.write(file_input4.value)
        stock_file.seek(0)
    return stock_file

@pn.depends(button7_Cu_A.param.clicks)
def calculate_A_batch_Cu(_):
    if csv_input_Cu_A() == 'kr_test.csv':
        kr_test_data = pd.read_csv(csv_input_Cu_A())
        kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC", "Alkalinity"]] 

        df=kr_test_data

        script = """
        <script>
        if (document.readyState === "complete") {
        $('.example').DataTable();
        } else {
        $(document).ready(function () {
            $('.example').DataTable();
        })
        }
        document.oncontextmenu=function(){return false;}
        document.onselectstart=function(){return false;}
        document.ondragstart=function(){return false;}
        </script>
        """
        html = df.to_html(classes=['example', 'panel-df'])
        test=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

    else:
        kr_test_data = pd.read_csv(csv_input_Cu_A())
        kr_test_data2=kr_test_data.copy()
        if 'Dissolved Cu' in kr_test_data :
            kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC", "Alkalinity"]] 
            kr_test_X = kr_test_data.iloc[:,:6]

            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_a_cu(kr_test_input_x), (-1,))
            kr_test_reshape_NN=np.power(10, (np.log10(kr_test_reshape_NN.detach().numpy()) - 0.41) / 0.85)
            a = kr_test_reshape_NN
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data,a,left_index=True,right_index=True)
            a['RCR']=a['BLM-based PNECs'] / kr_test_data2['Dissolved Cu']

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))
            
        else:
            kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC", "Alkalinity"]] 

            kr_test_X = kr_test_data.iloc[:,:6]

            kr_test_input_x = torch.Tensor(kr_test_X.values)
            kr_test_reshape_NN = torch.reshape(model_a_cu(kr_test_input_x), (-1,))
            kr_test_reshape_NN=np.power(10, (np.log10(kr_test_reshape_NN.detach().numpy()) - 0.41) / 0.85)

            a = kr_test_reshape_NN
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data,a,left_index=True,right_index=True)

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

        # def file():
        #     sio=StringIO()
        #     df.to_csv(sio, encoding="EUC-KR")
        #     sio.seek(0)
        #     return sio

        # file_download=pn.widgets.FileDownload(callback=file, filename="Calculate_MODEL_data.csv",sizing_mode='fixed')
        test=pn.Column(table)

    return test
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
    
model_C_Cu = MLPModel_C_Cu()
model_C_Cu.load_state_dict(torch.load('cond_Cu_Kor00.pth'))
#model(train_input_x)
text_input25 = pn.widgets.TextInput(name='pH', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input26 = pn.widgets.TextInput(name='Electronical conductivity (㎲/㎝)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input27 = pn.widgets.TextInput(name='Dissolved organic carbon (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input28 = pn.widgets.TextInput(name='Dissolved Cu', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
widget_box5=pn.Column(pn.pane.Markdown('### Input variables'),pn.Row(text_input25, text_input26, text_input27, text_input28))
def calculate_C_Cu():
    if text_input25.value and text_input26.value and text_input27.value =='': 
        kr_test_reshape_NN = 0
    else :
        x=float(text_input26.value)
        y=float(text_input27.value)
        list=[float(text_input25.value),np.log10(x),np.log10(y)]
        data=np.array(list)
        kr_test_input_x = torch.Tensor(data)
        kr_test_reshape_NN = torch.reshape(model_C_Cu(kr_test_input_x), (-1,))
    return np.power(10,(np.log10(kr_test_reshape_NN.item())-0.41)/0.85)
output_C_Cu = pn.widgets.TextInput(name='Copper BLM-based PNECs',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button_C_Cu = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])
output_C_Cu_2 = pn.widgets.TextInput(name='RCR',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button_C_Cu_2 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])
def button_event_Cu_C(event):
    output_C_Cu.value= str(calculate_C_Cu())
button_C_Cu.on_click(button_event_Cu_C)

def button_event_C_Cu_2(event):
    x=float(text_input28.value)/calculate_C_Cu()
    output_C_Cu_2.value= str(x)
button_C_Cu_2.on_click(button_event_C_Cu_2)
main_Cu_C = pn.Column(widget_box5,pn.Column(pn.pane.Markdown('<br>'),pn.Row(pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output_C_Cu,button_C_Cu)),pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output_C_Cu_2,button_C_Cu_2)))))
file_input_C_Cu = pn.widgets.FileInput(accept='.csv,.json',sizing_mode='fixed', css_classes=['panel-widget'])
button9_C_Cu = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def csv_input_C_Cu():
    if file_input_C_Cu.value is None:
        stock_file = 'kr_test(log)_c_test.csv'
    else : 
        stock_file = BytesIO()
        stock_file.write(file_input_C_Cu.value)
        stock_file.seek(0)
    return stock_file

@pn.depends(button9_C_Cu.param.clicks)
def calculate_C_batch_Cu(_):
    if csv_input_C_Cu() == 'kr_test(log)_c_test.csv':
        kr_test_data = pd.read_csv(csv_input_C_Cu())
        kr_test_data = kr_test_data[["pH", "Cond", "DOC",]] 

        df=kr_test_data

        script = """
        <script>
        if (document.readyState === "complete") {
        $('.example').DataTable();
        } else {
        $(document).ready(function () {
            $('.example').DataTable();
        })
        }
        document.oncontextmenu=function(){return false;}
        document.onselectstart=function(){return false;}
        document.ondragstart=function(){return false;}
        </script>
        """
        html = df.to_html(classes=['example', 'panel-df'])
        test=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

    else:
        kr_test_data = pd.read_csv(csv_input_C_Cu())
        kr_test_data2=kr_test_data.copy()

        if 'Dissolved Cu' in kr_test_data :
            kr_test_data = kr_test_data[["pH", "Cond", "DOC",]] 
            kr_test_data3=kr_test_data.copy()
            kr_test_data['Cond'] = np.log10(kr_test_data['Cond'])
            kr_test_data['DOC'] = np.log10(kr_test_data['DOC'])
            kr_test_X = kr_test_data.iloc[:,:3]

            kr_test_input_x = torch.Tensor(kr_test_X.values)




            kr_test_reshape_NN = torch.reshape(model_C_Cu(kr_test_input_x), (-1,))
            kr_test_reshape_NN=np.power(10, (np.log10(kr_test_reshape_NN.detach().numpy()) - 0.41) / 0.85)
            a = kr_test_reshape_NN
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data3,a,left_index=True,right_index=True)
            a['RCR']=a['BLM-based PNECs'] / kr_test_data2['Dissolved Cu']

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

        else:
            kr_test_data = kr_test_data[["pH", "Cond", "DOC",]] 
            kr_test_data3=kr_test_data.copy()
            kr_test_data['Cond'] = np.log10(kr_test_data['Cond'])
            kr_test_data['DOC'] = np.log10(kr_test_data['DOC'])
            kr_test_X = kr_test_data.iloc[:,:3]

            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_C_Cu(kr_test_input_x), (-1,))
            kr_test_reshape_NN=np.power(10, (np.log10(kr_test_reshape_NN.detach().numpy()) - 0.41) / 0.85)
            a = kr_test_reshape_NN
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data3,a,left_index=True,right_index=True)

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

        # def file():
        #     sio=StringIO()
        #     df.to_csv(sio, encoding="EUC-KR")
        #     sio.seek(0)
        #     return sio

        # file_download=pn.widgets.FileDownload(callback=file, filename="Calculate_MODEL3_data.csv",sizing_mode='fixed')
        test=pn.Column(table)

    return test 
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
text_input33 = pn.widgets.TextInput(name='pH', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input34 = pn.widgets.TextInput(name='Ca (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input35 = pn.widgets.TextInput(name='Mg (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input36 = pn.widgets.TextInput(name='Na (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input37 = pn.widgets.TextInput(name='Alkalinity (㎎ CaCo3/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input38 = pn.widgets.TextInput(name='Dissolved organic carbon (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input39 = pn.widgets.TextInput(name='Dissolved Cu (㎍/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
widget_box7=pn.Column(pn.pane.Markdown('### Input variables'),pn.Column(pn.Row(text_input33, text_input34, text_input35),pn.Row(text_input36, text_input37, text_input38, text_input39)))
def calculate_A_Ni():
    if text_input33.value and text_input34.value and text_input35.value and text_input36.value and text_input37.value and text_input38.value =='': 
        kr_test_reshape_NN = 'NONE'
    else :
        list=[float(text_input33.value),float(text_input36.value),float(text_input35.value),float(text_input34.value),float(text_input38.value),float(text_input37.value)]
        data=np.array(list)
        kr_test_input_x = torch.Tensor(data)
        kr_test_reshape_NN = torch.reshape(model_A_Ni(kr_test_input_x), (-1,))
    return kr_test_reshape_NN.item()
output_A_Ni = pn.widgets.TextInput(name='Nickel BLM-based PNECs',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button_A_Ni = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])
output_A_Ni_2 = pn.widgets.TextInput(name='RCR',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button_A_Ni_2 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def button_event_A_Ni(event):
    output_A_Ni.value= str(calculate_A_Ni())
button_A_Ni.on_click(button_event_A_Ni)

def button_event_A_Ni_2(event):
    x=float(text_input39.value)/calculate_A_Ni()
    output_A_Ni_2.value= str(x)
button_A_Ni_2.on_click(button_event_A_Ni_2)
main_Ni_A = pn.Column(widget_box7,pn.Column(pn.pane.Markdown('<br>'),pn.Row(pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output_A_Ni,button_A_Ni)),pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output_A_Ni_2,button_A_Ni_2)))))
file_input_A_Ni= pn.widgets.FileInput(accept='.csv,.json',sizing_mode='fixed', css_classes=['panel-widget'])
button9_A_Ni= pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def csv_input_A_Ni():
    if file_input_A_Ni.value is None:
        stock_file = 'kr_test.csv'
    else : 
        stock_file = BytesIO()
        stock_file.write(file_input_A_Ni.value)
        stock_file.seek(0)
    return stock_file

@pn.depends(button9_A_Ni.param.clicks)
def calculate_A_batch_Ni(_):
    if csv_input_A_Ni() == 'kr_test.csv':
        kr_test_data = pd.read_csv(csv_input_A_Ni())
        kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC", "Alkalinity"]] 

        df=kr_test_data

        script = """
        <script>
        if (document.readyState === "complete") {
        $('.example').DataTable();
        } else {
        $(document).ready(function () {
            $('.example').DataTable();
        })
        }
        document.oncontextmenu=function(){return false;}
        document.onselectstart=function(){return false;}
        document.ondragstart=function(){return false;}
        </script>
        """
        html = df.to_html(classes=['example', 'panel-df'])
        test=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

    else:
        kr_test_data = pd.read_csv(csv_input_A_Ni())
        kr_test_data2=kr_test_data.copy()
        if 'Dissolved Cu' in kr_test_data :
            kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC", "Alkalinity"]] 
            kr_test_X = kr_test_data.iloc[:,:6]

            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_A_Ni(kr_test_input_x), (-1,))
            a = kr_test_reshape_NN.detach().numpy()
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data,a,left_index=True,right_index=True)
            a['RCR']=a['BLM-based PNECs'] / kr_test_data2['Dissolved Cu']

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))
            
        else:
            kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC", "Alkalinity"]] 

            kr_test_X = kr_test_data.iloc[:,:6]

            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_A_Ni(kr_test_input_x), (-1,))
            a = kr_test_reshape_NN.detach().numpy()
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data,a,left_index=True,right_index=True)

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

        # def file():
        #     sio=StringIO()
        #     df.to_csv(sio, encoding="EUC-KR")
        #     sio.seek(0)
        #     return sio

        # file_download=pn.widgets.FileDownload(callback=file, filename="Calculate_MODEL_data.csv",sizing_mode='fixed')
        test=pn.Column(table)

    return test
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
text_input29 = pn.widgets.TextInput(name='pH', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input30 = pn.widgets.TextInput(name='Electronical conductivity (㎲/㎝)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input31 = pn.widgets.TextInput(name='Dissolved organic carbon (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input32 = pn.widgets.TextInput(name='Dissolved Cu', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
widget_box6=pn.Column(pn.pane.Markdown('### Input variables'),pn.Row(text_input29, text_input30, text_input31, text_input32))
def calculate_C_Ni():
    if text_input29.value and text_input30.value and text_input31.value =='': 
        kr_test_reshape_NN = 0
    else :
        x=float(text_input30.value)
        y=float(text_input31.value)
        P_Ak= -1.954980 +float(text_input29.value) * 0.161542 + np.log10(y) * 0.119819 + np.log10(x)  * 0.988312
        list=[float(text_input29.value),np.log10(x),np.log10(y),P_Ak]
        data=np.array(list)
        kr_test_input_x = torch.Tensor(data)
        kr_test_reshape_NN = torch.reshape(model_C_Ni(kr_test_input_x), (-1,))
    return np.power(10,kr_test_reshape_NN.item())
output_C_Ni = pn.widgets.TextInput(name='Nickel BLM-based PNECs',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button_C_Ni = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])
output_C_Ni_2 = pn.widgets.TextInput(name='RCR',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button_C_Ni_2 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def button_event_Ni_C(event):
    output_C_Ni.value= str(calculate_C_Ni())
button_C_Ni.on_click(button_event_Ni_C)

def button_event_C_Ni_2(event):
    x=float(text_input32.value)/calculate_C_Ni()
    output_C_Ni_2.value= str(x)
button_C_Ni_2.on_click(button_event_C_Ni_2)
main_Ni_C = pn.Column(widget_box6,pn.Column(pn.pane.Markdown('<br>'),pn.Row(pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output_C_Ni,button_C_Ni)),pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output_C_Ni_2,button_C_Ni_2)))))
file_input_C_Ni = pn.widgets.FileInput(accept='.csv,.json',sizing_mode='fixed', css_classes=['panel-widget'])
button9_C_Ni = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def csv_input_C_Ni():
    if file_input_C_Ni.value is None:
        stock_file = 'kr_test(log)_c_test.csv'
    else : 
        stock_file = BytesIO()
        stock_file.write(file_input_C_Ni.value)
        stock_file.seek(0)
    return stock_file

@pn.depends(button9_C_Ni.param.clicks)
def calculate_C_batch_Ni(_):
    if csv_input_C_Ni() == 'kr_test(log)_c_test.csv':
        kr_test_data = pd.read_csv(csv_input_C_Ni())
        kr_test_data = kr_test_data[["pH", "Cond", "DOC",]] 

        df=kr_test_data

        script = """
        <script>
        if (document.readyState === "complete") {
        $('.example').DataTable();
        } else {
        $(document).ready(function () {
            $('.example').DataTable();
        })
        }
        document.oncontextmenu=function(){return false;}
        document.onselectstart=function(){return false;}
        document.ondragstart=function(){return false;}
        </script>
        """
        html = df.to_html(classes=['example', 'panel-df'])
        test=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

    else:
        kr_test_data = pd.read_csv(csv_input_C_Ni())
        kr_test_data2=kr_test_data.copy()

        if 'Dissolved Cu' in kr_test_data :
            kr_test_data = kr_test_data[["pH", "Cond", "DOC"]] 
            kr_test_data3=kr_test_data.copy()
            kr_test_data['Cond'] = np.log10(kr_test_data['Cond'])
            kr_test_data['DOC'] = np.log10(kr_test_data['DOC'])
            P_Ak= -1.954980 +kr_test_data['pH'] * 0.161542 + kr_test_data['DOC'] * 0.119819 + kr_test_data['Cond']  * 0.988312   
            P_Alkalinity = np.power(10, P_Ak)
            kr_test_data['P_Alkalinity'] = P_Alkalinity 
            kr_test_data['P_Alkalinity'] = np.log10(kr_test_data['P_Alkalinity'])

            kr_test_X = kr_test_data.iloc[:,:4]
            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_C_Ni(kr_test_input_x), (-1,))
            a = np.power(10,kr_test_reshape_NN.detach().numpy())
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data3,a,left_index=True,right_index=True)
            a['RCR']=a['BLM-based PNECs'] / kr_test_data2['Dissolved Cu']

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

        else:
            kr_test_data = kr_test_data[["pH", "Cond", "DOC"]] 
            kr_test_data3=kr_test_data.copy()
            kr_test_data['Cond'] = np.log10(kr_test_data['Cond'])
            kr_test_data['DOC'] = np.log10(kr_test_data['DOC'])
            P_Ak= -1.954980 +kr_test_data['pH'] * 0.161542 + kr_test_data['DOC'] * 0.119819 + kr_test_data['Cond']  * 0.988312   
            P_Alkalinity = np.power(10, P_Ak)
            kr_test_data['P_Alkalinity'] = P_Alkalinity 
            kr_test_data['P_Alkalinity'] = np.log10(kr_test_data['P_Alkalinity'])

            kr_test_X = kr_test_data.iloc[:,:4]
            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_C_Ni(kr_test_input_x), (-1,))
            a = np.power(10,kr_test_reshape_NN.detach().numpy())
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data3,a,left_index=True,right_index=True)

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

        # def file():
        #     sio=StringIO()
        #     df.to_csv(sio, encoding="EUC-KR")
        #     sio.seek(0)
        #     return sio

        # file_download=pn.widgets.FileDownload(callback=file, filename="Calculate_MODEL3_data.csv",sizing_mode='fixed')
        test=pn.Column(table)

    return test 
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
#model(train_input_x)
text_input40 = pn.widgets.TextInput(name='pH', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input41 = pn.widgets.TextInput(name='Ca (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input42 = pn.widgets.TextInput(name='Mg (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input43 = pn.widgets.TextInput(name='Na (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input44 = pn.widgets.TextInput(name='Alkalinity (㎎ CaCo3/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input45 = pn.widgets.TextInput(name='Dissolved organic carbon (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input46 = pn.widgets.TextInput(name='Dissolved Cu (㎍/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
widget_box8=pn.Column(pn.pane.Markdown('### Input variables'),pn.Column(pn.Row(text_input40, text_input41, text_input42),pn.Row(text_input43, text_input44, text_input45, text_input46)))
def calculate_A_Zn():
    if text_input40.value and text_input41.value and text_input42.value and text_input43.value and text_input44.value and text_input45.value =='': 
        kr_test_reshape_NN = 'NONE'
    else :
        list=[float(text_input40.value),float(text_input43.value),float(text_input42.value),float(text_input41.value),float(text_input45.value),float(text_input44.value)]
        data=np.array(list)
        kr_test_input_x = torch.Tensor(data)
        kr_test_reshape_NN = torch.reshape(model_A_Zn(kr_test_input_x), (-1,))
    return kr_test_reshape_NN.item()
output_A_Zn = pn.widgets.TextInput(name='Zinc BLM-based PNECs',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button_A_Zn = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])
output_A_Zn_2 = pn.widgets.TextInput(name='RCR',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button_A_Zn_2 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def button_event_A_Zn(event):
    output_A_Zn.value= str(calculate_A_Zn())
button_A_Zn.on_click(button_event_A_Zn)

def button_event_A_Zn_2(event):
    x=float(text_input46.value)/calculate_A_Zn()
    output_A_Zn_2.value= str(x)
button_A_Zn_2.on_click(button_event_A_Zn_2)
main_A_Zn = pn.Column(widget_box8,pn.Column(pn.pane.Markdown('<br>'),pn.Row(pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output_A_Zn,button_A_Zn)),pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output_A_Zn_2,button_A_Zn_2)))))
file_input_A_Zn= pn.widgets.FileInput(accept='.csv,.json',sizing_mode='fixed', css_classes=['panel-widget'])
button9_A_Zn= pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def csv_input_A_Zn():
    if file_input_A_Zn.value is None:
        stock_file = 'kr_test.csv'
    else : 
        stock_file = BytesIO()
        stock_file.write(file_input_A_Zn.value)
        stock_file.seek(0)
    return stock_file

@pn.depends(button9_A_Zn.param.clicks)
def calculate_A_batch_Zn(_):
    if csv_input_A_Zn() == 'kr_test.csv':
        kr_test_data = pd.read_csv(csv_input_A_Zn())
        kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC", "Alkalinity"]] 

        df=kr_test_data

        script = """
        <script>
        if (document.readyState === "complete") {
        $('.example').DataTable();
        } else {
        $(document).ready(function () {
            $('.example').DataTable();
        })
        }
        document.oncontextmenu=function(){return false;}
        document.onselectstart=function(){return false;}
        document.ondragstart=function(){return false;}
        </script>
        """
        html = df.to_html(classes=['example', 'panel-df'])
        test=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

    else:
        kr_test_data = pd.read_csv(csv_input_A_Zn())
        kr_test_data2=kr_test_data.copy()
        if 'Dissolved Cu' in kr_test_data :
            kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC", "Alkalinity"]] 
            kr_test_X = kr_test_data.iloc[:,:6]

            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_A_Zn(kr_test_input_x), (-1,))
            a = kr_test_reshape_NN.detach().numpy()
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data,a,left_index=True,right_index=True)
            a['RCR']=a['BLM-based PNECs'] / kr_test_data2['Dissolved Cu']

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))
            
        else:
            kr_test_data = kr_test_data[["pH", "Na", "Mg", "Ca", "DOC", "Alkalinity"]] 

            kr_test_X = kr_test_data.iloc[:,:6]

            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_A_Zn(kr_test_input_x), (-1,))
            a = kr_test_reshape_NN.detach().numpy()
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data,a,left_index=True,right_index=True)

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

        # def file():
        #     sio=StringIO()
        #     df.to_csv(sio, encoding="EUC-KR")
        #     sio.seek(0)
        #     return sio

        # file_download=pn.widgets.FileDownload(callback=file, filename="Calculate_MODEL_data.csv",sizing_mode='fixed')
        test=pn.Column(table)

    return test
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
#model(train_input_x)
text_input47 = pn.widgets.TextInput(name='pH', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input48 = pn.widgets.TextInput(name='Electronical conductivity (㎲/㎝)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input49 = pn.widgets.TextInput(name='Dissolved organic carbon (㎎/L)', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
text_input50 = pn.widgets.TextInput(name='Dissolved Cu', placeholder='Enter a value here...',sizing_mode='fixed', css_classes=['panel-widget'])
widget_box9=pn.Column(pn.pane.Markdown('### Input variables'),pn.Row(text_input47, text_input48, text_input49, text_input50))
def calculate_C_Zn():
    if text_input47.value and text_input48.value and text_input49.value =='': 
        kr_test_reshape_NN = 0
    else :
        x=float(text_input48.value)
        y=float(text_input49.value)
        list=[float(text_input47.value),np.log10(x),np.log10(y)]
        data=np.array(list)
        kr_test_input_x = torch.Tensor(data)
        kr_test_reshape_NN = torch.reshape(model_C_Zn(kr_test_input_x), (-1,))
    return kr_test_reshape_NN.item()
output_C_Zn = pn.widgets.TextInput(name='Zinc BLM-based PNECs',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button_C_Zn = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])
output_C_Zn_2 = pn.widgets.TextInput(name='RCR',value='',disabled=True,sizing_mode='fixed', css_classes=['panel-widget'])
button_C_Zn_2 = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def button_event_Zn_C(event):
    output_C_Zn.value= str(calculate_C_Zn())
button_C_Zn.on_click(button_event_Zn_C)

def button_event_C_Zn_2(event):
    x=float(text_input50.value)/calculate_C_Zn()
    output_C_Zn_2.value= str(x)
button_C_Zn_2.on_click(button_event_C_Zn_2)
main_Zn_C = pn.Column(widget_box9,pn.Column(pn.pane.Markdown('<br>'),pn.Row(pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output_C_Zn,button_C_Zn)),pn.Column(pn.pane.Markdown('### Output variables'),pn.Row(output_C_Zn_2,button_C_Zn_2)))))
file_input_C_Zn = pn.widgets.FileInput(accept='.csv,.json',sizing_mode='fixed', css_classes=['panel-widget'])
button9_C_Zn = pn.widgets.Button(name='Calculate', button_type='primary',sizing_mode='fixed', css_classes=['button'])

def csv_input_C_Zn():
    if file_input_C_Zn.value is None:
        stock_file = 'kr_test(log)_c_test.csv'
    else : 
        stock_file = BytesIO()
        stock_file.write(file_input_C_Zn.value)
        stock_file.seek(0)
    return stock_file

@pn.depends(button9_C_Zn.param.clicks)
def calculate_C_batch_Zn(_):
    if csv_input_C_Zn() == 'kr_test(log)_c_test.csv':
        kr_test_data = pd.read_csv(csv_input_C_Zn())
        kr_test_data = kr_test_data[["pH", "Cond", "DOC",]] 

        df=kr_test_data

        script = """
        <script>
        if (document.readyState === "complete") {
        $('.example').DataTable();
        } else {
        $(document).ready(function () {
            $('.example').DataTable();
        })
        }
        document.oncontextmenu=function(){return false;}
        document.onselectstart=function(){return false;}
        document.ondragstart=function(){return false;}
        </script>
        """
        html = df.to_html(classes=['example', 'panel-df'])
        test=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

    else:
        kr_test_data = pd.read_csv(csv_input_C_Zn())
        kr_test_data2=kr_test_data.copy()

        if 'Dissolved Cu' in kr_test_data :
            kr_test_data = kr_test_data[["pH", "Cond", "DOC"]] 
            kr_test_data3=kr_test_data.copy()
            kr_test_data['Cond'] = np.log10(kr_test_data['Cond'])
            kr_test_data['DOC'] = np.log10(kr_test_data['DOC'])

            kr_test_X = kr_test_data.iloc[:,:3]
            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_C_Zn(kr_test_input_x), (-1,))
            a = kr_test_reshape_NN.detach().numpy()
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data3,a,left_index=True,right_index=True)
            a['RCR']=a['BLM-based PNECs'] / kr_test_data2['Dissolved Cu']

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

        else:
            kr_test_data = kr_test_data[["pH", "Cond", "DOC"]] 
            kr_test_data3=kr_test_data.copy()
            kr_test_data['Cond'] = np.log10(kr_test_data['Cond'])
            kr_test_data['DOC'] = np.log10(kr_test_data['DOC'])

            kr_test_X = kr_test_data.iloc[:,:3]
            kr_test_input_x = torch.Tensor(kr_test_X.values)

            kr_test_reshape_NN = torch.reshape(model_C_Zn(kr_test_input_x), (-1,))
            a = kr_test_reshape_NN.detach().numpy()
            a = pd.DataFrame(a)
            a.columns=['BLM-based PNECs']
            a=pd.merge(kr_test_data3,a,left_index=True,right_index=True)

            df=a
            script = """
            <script>
            if (document.readyState === "complete") {
            $('.example').DataTable();
            } else {
            $(document).ready(function () {
                $('.example').DataTable();
            })
            }
            document.oncontextmenu=function(){return false;}
            document.onselectstart=function(){return false;}
            document.ondragstart=function(){return false;}
            </script>
            """
            html = df.to_html(classes=['example', 'panel-df'])
            table=pn.Column(pn.pane.Markdown("Data frame"),pn.pane.HTML(html+script, sizing_mode='stretch_width'))

        # def file():
        #     sio=StringIO()
        #     df.to_csv(sio, encoding="EUC-KR")
        #     sio.seek(0)
        #     return sio

        # file_download=pn.widgets.FileDownload(callback=file, filename="Calculate_MODEL3_data.csv",sizing_mode='fixed')
        test=pn.Column(table)

    return test 
mmain1=pn.Column(mark3,pn.Row(file_input,button7),calculate_A_batch,mark)
mmain2=pn.Column(mark3,pn.Row(file_input_B,button8),calculate_B_batch,mark)
mmain3=pn.Column(mark3,pn.Row(file_input_C,button9),calculate_C_batch,mark)
mmain1_asia_cu_a=pn.Column(mark3,pn.Row(file_input4,button7_Cu_A),calculate_A_batch_Cu,mark)
mmain1_asia_cu_c=pn.Column(mark3,pn.Row(file_input_C_Cu,button9_C_Cu),calculate_C_batch_Cu,mark)
mmain1_asia_ni_a=pn.Column(mark3,pn.Row(file_input_A_Ni,button9_A_Ni),calculate_A_batch_Ni,mark)
mmain1_asia_ni_c=pn.Column(mark3,pn.Row(file_input_C_Ni,button9_C_Ni),calculate_C_batch_Ni,mark)
mmain1_asia_zn_a=pn.Column(mark3,pn.Row(file_input_A_Zn,button9_A_Zn),calculate_A_batch_Zn,mark)
mmain1_asia_zn_c=pn.Column(mark3,pn.Row(file_input_C_Zn,button9_C_Zn),calculate_C_batch_Zn,mark)
radio_group3 = pn.widgets.RadioButtonGroup(
    name='Radio Button Group', options=['Copper', 'Nickel', 'Zinc'], sizing_mode='stretch_width', button_type='primary',margin=(0,0,50,0),css_classes=['widget-button'])
radio_group4 = pn.widgets.RadioButtonGroup(
    name='Radio Button Group', options=['Copper'], sizing_mode='stretch_width', button_type='primary',margin=(0,0,50,0),css_classes=['widget-button'])
@pn.depends(x=radio_group3.param.value)
def main_s(x):
    if x =='Copper':
        tab= pn.Tabs(
            ("Copper_DNN(a)",pn.Column(main_Cu_A,mark2,mmain1_asia_cu_a)),
            ("Copper_DNN(c)",pn.Column(main_Cu_C,mark2,mmain1_asia_cu_c)),
            dynamic=True
        )
    elif x =='Nickel':
        tab= pn.Tabs(
            ("Nickel_DNN(a)",pn.Column(main_Ni_A,mark2,mmain1_asia_ni_a)),
            ("Nickel_DNN(c)",pn.Column(main_Ni_C,mark2,mmain1_asia_ni_c)),
            dynamic=True
        )
    elif x =='Zinc':
        tab= pn.Tabs(
            ("Zinc_DNN(a)",pn.Column(main_A_Zn,mark2,mmain1_asia_zn_a)),
            ("Zinc_DNN(c)",pn.Column(main_Zn_C,mark2,mmain1_asia_zn_c)),
            dynamic=True
        )
    return pn.Column(radio_group3,tab)
@pn.depends(x=select_main.param.value)
def select_mains(x):
    if x=='MAIN':
        dnn_tab=pn.Column(pn.pane.JPG('main.jpg',height=803,width=1130,margin=(0,0,50,0)))
    elif x=='European species':
        dnn_tabs = pn.Tabs(
            ("DNN(a)",pn.Column(main,mark2,mmain1)),
            ("DNN(b)",pn.Column(main2,mark2,mmain2)),
            ("DNN(c)",pn.Column(main3,mark2,mmain3)),
            dynamic=True
        )
        dnn_tab=pn.Column(radio_group4,dnn_tabs)
    elif x=='Asian species':
        dnn_tab =pn.Column(main_s)
    return dnn_tab
template = pn.template.MaterialTemplate(
    site="EHR&C", title="DNN model for BLM vol.1" ,
    sidebar=[select_main],
    main=[select_mains],
)


template.sidebar_width=400
template.servable()