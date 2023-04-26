import json
from flask import Flask, request, jsonify
import cu_model
import pandas as pd
import numpy as np
import random
import cu_model_asia

app = Flask(__name__)

@app.route('/cu', methods=['POST'])
def test():
    data = request.get_json() 
    # # print(data)
    cu_ph=data['cu_pH_input']
    cu_mg=data['cu_mg_input']
    cu_ca=data['cu_ca_input']
    cu_na=data['cu_na_input']
    cu_al=data['cu_al_input']
    cu_doc=data['cu_doc_input']
    cu_dc=data['cu_dc_input'] #eu 
    cu_dc2=data['cu_dc2_input'] #asia
    
    if cu_dc2 =="":
        result,result2=cu_model.eu_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc)
        return json.dumps({"result":result,"result2":result2})
    elif cu_dc=="":
        result3,result4=cu_model.asia_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc2)
        return json.dumps({"result3":result3,"result4":result4})
    else :
        result,result2=cu_model.eu_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc)
        result3,result4=cu_model.asia_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc2)
        return json.dumps({"result":result,"result2":result2,"result3":result3,"result4":result4})

@app.route('/cu2', methods=['POST'])
def test_2():
    # NODE.JS -> FLASK CSV FILE SAVE
    df = pd.read_csv(request.files['file'] ,encoding='CP949')
    df_e=cu_model.eu_a_csv(df)
    df_a=cu_model.asia_a_csv(df)
    df_s=pd.concat([df_e,df_a],axis=1)
    df_s=df_s.head(15)
    df_s.to_csv("all.csv")

    # if('Chronic RCR' in df_s.columns & 'BLM-based chronic PNEC'in df_s.columns):
    if('Acute RCR' in df_s.columns and 'Chronic RCR' in df_s.columns ):    
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),"Alkalinity":df_s['Alkalinity (㎎ CaCO3/L)'].tolist(),
                    "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":df_s['Chronic RCR'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist(),"Acute RCR":df_s['Acute RCR'].tolist()})
    elif('Chronic RCR' in df_s.columns):
        # if (df_s['Chronic RCR'].isnull().all()):
        #     chronic_rcr = [0] * len(df_s['pH'])
        #     print(chronic_rcr)
        #     return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),"Alkalinity":df_s['Alkalinity (㎎ CaCO3/L)'].tolist(),
        #         "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":chronic_rcr,"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})
        # else:
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),"Alkalinity":df_s['Alkalinity (㎎ CaCO3/L)'].tolist(),
                        "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":df_s['Chronic RCR'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})
    elif('Acute RCR' in df_s.columns):
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),"Alkalinity":df_s['Alkalinity (㎎ CaCO3/L)'].tolist(),
                           "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist(),"Acute RCR":df_s['Acute RCR'].tolist()})
    else:
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),"Alkalinity":df_s['Alkalinity (㎎ CaCO3/L)'].tolist(),
                           "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})
    
@app.route('/cu3', methods=['POST'])
def test_3():
    data = request.get_json() 
    # print(data)
    cu_ph=data['cu_pH_input']
    cu_mg=data['cu_mg_input']
    cu_ca=data['cu_ca_input']
    cu_na=data['cu_na_input']
    cu_doc=data['cu_doc_input']
    cu_dc=data['cu_dc_input']
    cu_dc2=data['cu_dc2_input'] #asia

    if cu_dc2 =="":
        result,result2=cu_model.eu_b(cu_ph,cu_na,cu_ca,cu_mg,cu_doc,cu_dc)
        return json.dumps({"result":result,"result2":result2})
    elif cu_dc =="":
        result3,result4=cu_model.asia_b(cu_ph,cu_na,cu_ca,cu_mg,cu_doc,cu_dc2)
        return json.dumps({"result3":result3,"result4":result4})
    else:
        result,result2=cu_model.eu_b(cu_ph,cu_na,cu_ca,cu_mg,cu_doc,cu_dc)
        result3,result4=cu_model.asia_b(cu_ph,cu_na,cu_ca,cu_mg,cu_doc,cu_dc2)
        return json.dumps({"result":result,"result2":result2,"result3":result3,"result4":result4})


@app.route('/cu4', methods=['POST'])
def test_4():
    # NODE.JS -> FLASK CSV FILE SAVE
    df = pd.read_csv(request.files['file'],encoding='CP949')
    df_e=cu_model.eu_b_csv(df)
    df_a=cu_model.asia_b_csv(df)
    df_s=pd.concat([df_e,df_a],axis=1)
    # df_s.to_csv("all.csv")
    df_s=df_s.head(15)

    if('Acute RCR' in df_s.columns and 'Chronic RCR' in df_s.columns ):   
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                           "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":df_s['Chronic RCR'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist(),"Acute RCR":df_s['Acute RCR'].tolist()})
    elif('Chronic RCR' in df_s.columns):
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                           "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":df_s['Chronic RCR'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})
    elif('Acute RCR' in df_s.columns):
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                    "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist(),"Acute RCR":df_s['Acute RCR'].tolist()})
    else:
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})

@app.route('/cu5', methods=['POST'])
def test_5():
    data = request.get_json() 
    # print(data)
    cu_ph=data['cu_pH_input']
    cu_ec=data['cu_ec_input']
    cu_doc=data['cu_doc_input']
    cu_dc=data['cu_dc_input']
    cu_dc2=data['cu_dc2_input'] #asia

    if cu_dc2=="":
        result,result2=cu_model.eu_c(cu_ph,cu_ec,cu_doc,cu_dc)
        return json.dumps({"result":result,"result2":result2})
    elif cu_dc=="":
        result3,result4=cu_model.asia_c(cu_ph,cu_ec,cu_doc,cu_dc2)
        return json.dumps({"result3":result3,"result4":result4})
    else:
        result,result2=cu_model.eu_c(cu_ph,cu_ec,cu_doc,cu_dc)
        result3,result4=cu_model.asia_c(cu_ph,cu_ec,cu_doc,cu_dc2)
        return json.dumps({"result":result,"result2":result2,"result3":result3,"result4":result4})

@app.route('/cu6', methods=['POST'])
def test_6():
    # NODE.JS -> FLASK CSV FILE SAVE
    df = pd.read_csv(request.files['file'],encoding='CP949')
    df_e=cu_model.eu_c_csv(df)
    df_a=cu_model.asia_c_csv(df)
    df_s=pd.concat([df_e,df_a],axis=1)
    # df_s.to_csv("all_c.csv")
    df_s=df_s.head(15)

#    print(df_s)

    if('Acute RCR' in df_s.columns and 'Chronic RCR' in df_s.columns ):   
        return json.dumps({"pH":df_s['pH'].tolist(),"Cond":df_s['Cond (㎲/㎝)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                           "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":df_s['Chronic RCR'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist(),"Acute RCR":df_s['Acute RCR'].tolist()})
    elif('Chronic RCR' in df_s.columns):
        return json.dumps({"pH":df_s['pH'].tolist(),"Cond":df_s['Cond (㎲/㎝)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                           "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":df_s['Chronic RCR'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})
    elif('Acute RCR' in df_s.columns):
        return json.dumps({"pH":df_s['pH'].tolist(),"Cond":df_s['Cond (㎲/㎝)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                    "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist(),"Acute RCR":df_s['Acute RCR'].tolist()})
    else:
        return json.dumps({"pH":df_s['pH'].tolist(),"Cond":df_s['Cond (㎲/㎝)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})


@app.route('/cu_asia', methods=['POST'])
def test_asia():
    data = request.get_json() 
    # print(data)
    cu_ph=data['cu_pH_input']
    cu_mg=data['cu_mg_input']
    cu_ca=data['cu_ca_input']
    cu_na=data['cu_na_input']
    cu_al=data['cu_al_input']
    cu_doc=data['cu_doc_input']
    cu_dc=data['cu_dc_input'] #eu 
    cu_dc2=data['cu_dc2_input'] #asia
    
    if cu_dc2 =="":
        result,result2=cu_model_asia.eu_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc)
        return json.dumps({"result":result,"result2":result2})
    elif cu_dc=="":
        result3,result4=cu_model_asia.asia_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc2)
        return json.dumps({"result3":result3,"result4":result4})
    else :
        result,result2=cu_model_asia.eu_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc)
        result3,result4=cu_model_asia.asia_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc2)
        return json.dumps({"result":result,"result2":result2,"result3":result3,"result4":result4})

@app.route('/cu2_asia', methods=['POST'])
def test_2_asia():
    # NODE.JS -> FLASK CSV FILE SAVE
    df = pd.read_csv(request.files['file'],encoding='CP949')
    df_e=cu_model_asia.eu_a_csv(df)
    df_a=cu_model_asia.asia_a_csv(df)
    df_s=pd.concat([df_e,df_a],axis=1)
    df_s=df_s.head(15)
    df_s.to_csv("all.csv")

    # if('Chronic RCR' in df_s.columns & 'BLM-based chronic PNEC'in df_s.columns):
    if('Acute RCR' in df_s.columns and 'Chronic RCR' in df_s.columns ):    
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),"Alkalinity":df_s['Alkalinity (㎎ CaCO3/L)'].tolist(),
                    "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":df_s['Chronic RCR'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist(),"Acute RCR":df_s['Acute RCR'].tolist()})
    elif('Chronic RCR' in df_s.columns):
        # if (df_s['Chronic RCR'].isnull().all()):
        #     chronic_rcr = [0] * len(df_s['pH'])
        #     print(chronic_rcr)
        #     return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),"Alkalinity":df_s['Alkalinity (㎎ CaCO3/L)'].tolist(),
        #         "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":chronic_rcr,"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})
        # else:
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),"Alkalinity":df_s['Alkalinity (㎎ CaCO3/L)'].tolist(),
                        "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":df_s['Chronic RCR'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})
    elif('Acute RCR' in df_s.columns):
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),"Alkalinity":df_s['Alkalinity (㎎ CaCO3/L)'].tolist(),
                           "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist(),"Acute RCR":df_s['Acute RCR'].tolist()})
    else:
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),"Alkalinity":df_s['Alkalinity (㎎ CaCO3/L)'].tolist(),
                           "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})
    
@app.route('/cu3_asia', methods=['POST'])
def test_3_asia():
    data = request.get_json() 
    # print(data)
    cu_ph=data['cu_pH_input']
    cu_mg=data['cu_mg_input']
    cu_ca=data['cu_ca_input']
    cu_na=data['cu_na_input']
    cu_doc=data['cu_doc_input']
    cu_dc=data['cu_dc_input']
    cu_dc2=data['cu_dc2_input'] #asia

    if cu_dc2 =="":
        result,result2=cu_model_asia.eu_b(cu_ph,cu_na,cu_ca,cu_mg,cu_doc,cu_dc)
        return json.dumps({"result":result,"result2":result2})
    elif cu_dc =="":
        result3,result4=cu_model_asia.asia_b(cu_ph,cu_na,cu_ca,cu_mg,cu_doc,cu_dc2)
        return json.dumps({"result3":result3,"result4":result4})
    else:
        result,result2=cu_model_asia.eu_b(cu_ph,cu_na,cu_ca,cu_mg,cu_doc,cu_dc)
        result3,result4=cu_model_asia.asia_b(cu_ph,cu_na,cu_ca,cu_mg,cu_doc,cu_dc2)
        return json.dumps({"result":result,"result2":result2,"result3":result3,"result4":result4})


@app.route('/cu4_asia', methods=['POST'])
def test_4_asia():
    # NODE.JS -> FLASK CSV FILE SAVE
    df = pd.read_csv(request.files['file'],encoding='CP949')
    df_e=cu_model_asia.eu_b_csv(df)
    df_a=cu_model_asia.asia_b_csv(df)
    df_s=pd.concat([df_e,df_a],axis=1)
    # df_s.to_csv("all.csv")
    df_s=df_s.head(15)

    if('Acute RCR' in df_s.columns and 'Chronic RCR' in df_s.columns ):   
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                           "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":df_s['Chronic RCR'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist(),"Acute RCR":df_s['Acute RCR'].tolist()})
    elif('Chronic RCR' in df_s.columns):
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                           "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":df_s['Chronic RCR'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})
    elif('Acute RCR' in df_s.columns):
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                    "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist(),"Acute RCR":df_s['Acute RCR'].tolist()})
    else:
        return json.dumps({"pH":df_s['pH'].tolist(),"Na":df_s['Na (mg/L)'].tolist(),"Mg":df_s['Mg (mg/L)'].tolist(),"Ca":df_s['Ca (mg/L)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})

@app.route('/cu5_asia', methods=['POST'])
def test_5_asia():
    data = request.get_json() 
    # print(data)
    cu_ph=data['cu_pH_input']
    cu_ec=data['cu_ec_input']
    cu_doc=data['cu_doc_input']
    cu_dc=data['cu_dc_input']
    cu_dc2=data['cu_dc2_input'] #asia

    if cu_dc2=="":
        result,result2=cu_model_asia.eu_c(cu_ph,cu_ec,cu_doc,cu_dc)
        return json.dumps({"result":result,"result2":result2})
    elif cu_dc=="":
        result3,result4=cu_model_asia.asia_c(cu_ph,cu_ec,cu_doc,cu_dc2)
        return json.dumps({"result3":result3,"result4":result4})
    else:
        result,result2=cu_model_asia.eu_c(cu_ph,cu_ec,cu_doc,cu_dc)
        result3,result4=cu_model_asia.asia_c(cu_ph,cu_ec,cu_doc,cu_dc2)
        return json.dumps({"result":result,"result2":result2,"result3":result3,"result4":result4})

@app.route('/cu6_asia', methods=['POST'])
def test_6_asia():
    # NODE.JS -> FLASK CSV FILE SAVE
    df = pd.read_csv(request.files['file'],encoding='CP949')
    df_e=cu_model_asia.eu_c_csv(df)
    df_a=cu_model_asia.asia_c_csv(df)
    df_s=pd.concat([df_e,df_a],axis=1)
    # df_s.to_csv("all_c.csv")
    df_s=df_s.head(15)

#    print(df_s)

    if('Acute RCR' in df_s.columns and 'Chronic RCR' in df_s.columns ):   
        return json.dumps({"pH":df_s['pH'].tolist(),"Cond":df_s['Cond (㎲/㎝)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                           "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":df_s['Chronic RCR'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist(),"Acute RCR":df_s['Acute RCR'].tolist()})
    elif('Chronic RCR' in df_s.columns):
        return json.dumps({"pH":df_s['pH'].tolist(),"Cond":df_s['Cond (㎲/㎝)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                           "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"Chronic RCR":df_s['Chronic RCR'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})
    elif('Acute RCR' in df_s.columns):
        return json.dumps({"pH":df_s['pH'].tolist(),"Cond":df_s['Cond (㎲/㎝)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                    "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist(),"Acute RCR":df_s['Acute RCR'].tolist()})
    else:
        return json.dumps({"pH":df_s['pH'].tolist(),"Cond":df_s['Cond (㎲/㎝)'].tolist(),"DOC":df_s['DOC (mg/L)'].tolist(),
                "BLM-based chronic PNEC":df_s['BLM-based chronic PNEC'].tolist(),"BLM-based acute PNEC":df_s['BLM-based acute PNEC'].tolist()})




# @app.route('/ni2', methods=['POST'])
# def test_4():
#     data = request.get_json() 
#     # print(data)
#     cu_ph=data['cu_pH_input']
#     cu_mg=data['cu_mg_input']
#     cu_ca=data['cu_ca_input']
#     cu_na=data['cu_na_input']
#     cu_al=data['cu_al_input']
#     cu_doc=data['cu_doc_input']
#     cu_dc=data['cu_dc_input']

#     result,result2=cu_model.eu_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc)
#     result3,result4=cu_model.asia_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc)
#     return json.dumps({"result":result,"result2":result2,"result3":result3,"result4":result4})

# @app.route('/zn', methods=['POST'])
# def test_5():
#     data = request.get_json() 
#     # print(data)
#     cu_ph=data['cu_pH_input']
#     cu_mg=data['cu_mg_input']
#     cu_ca=data['cu_ca_input']
#     cu_na=data['cu_na_input']
#     cu_al=data['cu_al_input']
#     cu_doc=data['cu_doc_input']
#     cu_dc=data['cu_dc_input']

#     result,result2=cu_model.eu_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc)
#     result3,result4=cu_model.asia_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc)
#     return json.dumps({"result":result,"result2":result2,"result3":result3,"result4":result4})

# @app.route('/zn2', methods=['POST'])
# def test_6():
#     data = request.get_json() 
#     # print(data)
#     cu_ph=data['cu_pH_input']
#     cu_mg=data['cu_mg_input']
#     cu_ca=data['cu_ca_input']
#     cu_na=data['cu_na_input']
#     cu_al=data['cu_al_input']
#     cu_doc=data['cu_doc_input']
#     cu_dc=data['cu_dc_input']

#     result,result2=cu_model.eu_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc)
#     result3,result4=cu_model.asia_a(cu_ph,cu_na,cu_ca,cu_mg,cu_al,cu_doc,cu_dc)
#     return json.dumps({"result":result,"result2":result2,"result3":result3,"result4":result4})

if __name__ == '__main__':
    app.run(host='172.30.1.43', port=5008)
    # app.run(host='127.0.0.1', port=5008)