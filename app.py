# from flask import Flask, render_template, request
# import pickle
# import numpy as np
# import pandas as pd


# # Load the  model
# filename = 'rank_model.pkl'
# filename1= 'polynomial_transform.pkl'
# model = pickle.load(open(filename, 'rb'))
# model1=pickle.load(open(filename1,'rb'))


# filename_uni= 'poly_univ.pkl'
# filename_uni1='rank_model_univ.pkl'
# model_uni = pickle.load(open(filename_uni, 'rb'))
# model_uni1=pickle.load(open(filename_uni1,'rb'))

# filename_over='poly_overall.pkl'
# filename_over1='rank_model_overall.pkl'
# model_over = pickle.load(open(filename_over, 'rb'))
# model_over1=pickle.load(open(filename_over1,'rb'))


# class modify:
#     def find_range(self,num):
#         if num<=0:
#             return "1- 5"
#         if num-3<=0:
#             return "1 - "+str(int(num)+3)
#         if num <=25:
#             return str(int(num)-3)+" - "+str(int(num)+3)
#         if num>=25 and num<=75:
#             return str(int(num)-4)+" - "+str(int(num)+4)
#         if num>=75 and num<=136:
#             return str(int(num)-7)+" - "+str(int(num)+7)
#         else:
#             return str(int(num)-19)+" - "+str(int(num)+19)
# class modify1:
#     def find_range(self,num):
#         if num<=0:
#             return "2 (+/- 1)"
#         if num-3<=0 or num-4<=0:
#             return "3 (+/- 3)"
#         if num <=25:
#             return str(num)+" (+/- 3)"
#         if num>=25 and num<=75:
#             return str(num)+" (+/- 4) " 
#         if num>=75 and num<=139:
#             return str(num)+" (+/- 7) " 
#         if num>=140 and num<=149:
#             return str(num)+" (+/- 11) " 
#         else:
#             return str(num)+" (+/- 19) " 
        
#     def predict_rank_u(self,num):
#         if num<0:
#             return "1 (+/- 2)"
#         if num<10:
#             return str(num)+" (+/- 2)"
#         if num<30:
#             return str(num)+" (+/- 2)"
#         if num<40:
#             return str(num)+" (+/- 4)"
#         if num<50:
#             return str(num)+" (+/- 6)"
#         if num<60:
#             return str(num)+" (+/- 7)"
#         if num<70:
#             return str(num)+" (+/- 4)"
#         if num<80:
#             return str(num)+" (+/- 2)"
#         if num<90:
#             return str(num)+" (+/- 2)"
#         else:
#             return str(num)+" (+/- 2)"
        
#     def predict_rank_o(self,num):
#         if num<0:
#             return "1 +/- 2"
#         if num<10:
#             return str(num)+" (+/- 2)"
#         if num<30:
#             return str(num)+" (+/- 2)"
#         if num<40:
#             return str(num)+" (+/- 4)"
#         if num<50:
#             return str(num)+" (+/- 1)"
#         if num<60:
#             return str(num)+" (+/- 2)"
#         if num<70:
#             return str(num)+" (+/- 2)"
#         if num<80:
#             return str(num)+" (+/- 2)"
#         if num<90:
#             return str(num)+" (+/- 1)"
#         else:
#             return str(num)+" (+/- 7)"        

# rank_uni_para=[]
# rpc_uni_para=[]
# tlr_uni_para=[]
# go_uni_para=[]
# oi_uni_para=[]
# ppn_uni_para=[]


# rank_over_para=[]
# rpc_over_para=[]
# tlr_over_para=[]
# go_over_para=[]
# oi_over_para=[]
# ppn_over_para=[]


# rank_engg_para=[]
# rpc_engg_para=[]
# tlr_engg_para=[]
# go_engg_para=[]
# oi_engg_para=[]
# ppn_engg_para=[]


# def uni_clear():
#     rank_uni_para.clear()
#     rpc_uni_para.clear()
#     tlr_uni_para.clear()
#     go_uni_para.clear()
#     oi_uni_para.clear()
#     ppn_uni_para.clear()

    
    
# def over_clear():
#     rank_over_para.clear()
#     rpc_over_para.clear()
#     tlr_over_para.clear()
#     go_over_para.clear()
#     oi_over_para.clear()
#     ppn_over_para.clear()

# def engg_clear():
#     rank_engg_para.clear()
#     rpc_engg_para.clear()
#     tlr_engg_para.clear()
#     go_engg_para.clear()
#     oi_engg_para.clear()
#     ppn_engg_para.clear()    


# app = Flask(__name__)
# app.secret_key = 'secret!!'







# @app.route('/')
# def home():
# 	return render_template('index.html')

# @app.route('/engg/')
# def engg():
#     return render_template('engg.html')

# @app.route('/university/')
# def university():
#     return render_template('university.html')

# @app.route('/overall/')
# def overall():
#     return render_template('overall.html')

# @app.route('/index')
# def index():
#     engg_clear()
#     over_clear()
#     uni_clear()
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         uni_clear()
#         over_clear()
#         tlr = float(request.form['tlr'])
#         rpc = float(request.form['rpc'])
#         go = float(request.form['go'])
#         oi =float(request.form['oi'])
#         perception = float(request.form['perception'])
#         tlr1_high= 93.55
#         rpc1_high= 96.18
#         go1_high= 89.84
#         oi1_high= 68.5
#         perception1_high= 100
#         tlr_1=tlr/tlr1_high
#         rpc_1=rpc/rpc1_high
#         go_1=go/go1_high
#         oi_1=oi/oi1_high
#         perception_1=perception/perception1_high
#         score=(tlr_1*0.3)+(rpc_1*0.3)+(go_1*0.2)+(oi_1*0.1)+(perception_1*0.1)
#         data = np.array([score*100]).reshape(1,-1)
#         pre_prediction=model1.fit_transform(data)
#         my_prediction = model.predict(pre_prediction)
        
        
#         tlr_engg_para.append(float(tlr))
#         rpc_engg_para.append(float(rpc))
#         go_engg_para.append(float(go))
#         oi_engg_para.append(float(oi))
#         ppn_engg_para.append(float(perception))
        
        


        
#         lis2=[]
#         lis2.append(tlr_engg_para)
#         lis2.append(rpc_engg_para)
#         lis2.append(go_engg_para)
#         lis2.append(oi_engg_para)
#         lis2.append(ppn_engg_para)
#         #lis2.append(rank_engg_para)
        
#         numpy_array_engg=np.array(lis2).T
#         df_engg=pd.DataFrame(numpy_array_engg)
#         df_engg.columns=['TLR','RPC','GO','OI','PPN']
#         l=[]
#         for i in range(numpy_array_engg.shape[0]):
#             l.append(i+1)
#         df_engg.index=l            
        
        
        
#         x=my_prediction[0,0]
#         range1=modify1()
#         ranges=range1.find_range(int(x))
#         if x<=140:
#             rank_engg_para.append(ranges)
#             df_engg['RANK']=rank_engg_para
#             return render_template('engg.html', prediction="The predicted rank might be in range : " + ranges,tables=[df_engg.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])
#         else:
#             rank_engg_para.append(">150")
#             df_engg['RANK']=rank_engg_para            
#             return render_template('engg.html',prediction="The rank for this score is greater than 150",tables=[df_engg.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])
#         ##return render_template('result.html', prediction=data.shape)


# @app.route('/predictUni', methods=['POST'])
# def predictUni():
#     if request.method == 'POST':
#         over_clear()
#         engg_clear()
#         tlr_u = float(request.form['tlr_u'])
#         rpc_u = float(request.form['rpc_u'])
#         go_u = float(request.form['go_u'])
#         oi_u =float(request.form['oi_u'])
#         perception_u = float(request.form['perception_u'])
#         score_u=(tlr_u*0.3)+(rpc_u*0.3)+(go_u*0.2)+(oi_u*0.1)+(perception_u*0.1)
#         data_u = np.array([score_u]).reshape(1,-1)
#         pre_prediction_u=model_uni.fit_transform(data_u)
#         my_prediction_u=model_uni1.predict(pre_prediction_u)
#         x1=my_prediction_u[0,0]
        
#         range2=modify1()
#         ranges2=range2.predict_rank_u(int(x1))

                  
#         tlr_uni_para.append(float(tlr_u))
#         rpc_uni_para.append(float(rpc_u))
#         go_uni_para.append(float(go_u))
#         oi_uni_para.append(float(oi_u))
#         ppn_uni_para.append(float(perception_u))
        
#

        
#         lis=[]
#         lis.append(tlr_uni_para)
#         lis.append(rpc_uni_para)
#         lis.append(go_uni_para)
#         lis.append(oi_uni_para)
#         lis.append(ppn_uni_para)
#         #lis.append(rank_uni_para)
        
#         numpy_array_uni=np.array(lis).T
#         df=pd.DataFrame(numpy_array_uni)
#         df.columns=['TLR','RPC','GO','OI','PPN']
#         l=[]
#         for i in range(numpy_array_uni.shape[0]):
#             l.append(i+1)
#         df.index=l        
    

        
#         if x1<=140:
#             rank_uni_para.append(ranges2)
#             df['RANK']=rank_uni_para
#             return render_template('university.html', prediction="The predicted rank might be in range : " + ranges2,tables=[df.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])
#         else:
#             rank_uni_para.append(">150")
#             df['RANK']=rank_uni_para            
#             return render_template('university.html',prediction="The rank for this score is greater than 150",tables=[df.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])            
        



# @app.route('/predictOver', methods=['POST'])
# def predictOver():
#     if request.method == 'POST':
        
#         uni_clear()
#         engg_clear()
        
        
        
#         tlr_o = float(request.form['tlr_o'])
#         rpc_o = float(request.form['rpc_o'])
#         go_o = float(request.form['go_o'])
#         oi_o =float(request.form['oi_o'])
#         perception_o = float(request.form['perception_o'])
#         score_o=(tlr_o*0.3)+(rpc_o*0.3)+(go_o*0.2)+(oi_o*0.1)+(perception_o*0.1)
#         data_o = np.array([score_o]).reshape(1,-1)
#         pre_prediction_o=model_over.fit_transform(data_o)
#         my_prediction_o=model_over1.predict(pre_prediction_o)
#         x2=my_prediction_o[0,0]
        
#         range3=modify1()
#         ranges3=range3.predict_rank_o(int(x2))        
        
#         tlr_over_para.append(float(tlr_o))
#         rpc_over_para.append(float(rpc_o))
#         go_over_para.append(float(go_o))
#         oi_over_para.append(float(oi_o))
#         ppn_over_para.append(float(perception_o))
                


#         lis1=[]
#         lis1.append(tlr_over_para)
#         lis1.append(rpc_over_para)
#         lis1.append(go_over_para)
#         lis1.append(oi_over_para)
#         lis1.append(ppn_over_para)
#         '''lis1.append(rank_over_para)'''
        
#         numpy_array_over=np.array(lis1).T
#         df_over=pd.DataFrame(numpy_array_over)
#         df_over.columns=['TLR','RPC','GO','OI','PPN']
#         l=[]
#         for i in range(numpy_array_over.shape[0]):
#             l.append(i+1)
#         df_over.index=l
        

                
#         if x2<=140:
#             rank_over_para.append(ranges3)
#             df_over['RANK']=rank_over_para
#             return render_template('overall.html', prediction="The predicted rank might be in range : " + ranges3,tables=[df_over.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])
#         else:
#             rank_over_para.append(">150")
#             df_over['RANK']=rank_over_para            
#             return render_template('overall.html',prediction="The rank for this score is greater than 150",tables=[df_over.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])

# if __name__ == '__main__':
#	app.run(debug=True)
	
	
	
	
	
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


# Load the  model

#filename = 'rank_model.pkl'
#filename1= 'polynomial_transform.pkl'
filename = 'models/rank_model.pkl'
filename1= 'models/polynomial_transform.pkl'
model = pickle.load(open(filename, 'rb'))
model1=pickle.load(open(filename1,'rb'))


filename_uni= 'poly_univ.pkl'
filename_uni1='rank_model_univ.pkl'
model_uni = pickle.load(open(filename_uni, 'rb'))
model_uni1=pickle.load(open(filename_uni1,'rb'))

filename_over='poly_overall.pkl'
filename_over1='rank_model_overall.pkl'
model_over = pickle.load(open(filename_over, 'rb'))
model_over1=pickle.load(open(filename_over1,'rb'))



class modify1:
    def find_range(self,num):
        if num<=0:
            return "2 (+/- 1)"
        elif num<=100:
            return str(num)+" (+/- 3)"
        elif num <=120:
            return str(num)+" (+/- 3)"
        elif num<=140:
            return str(num)+" (+/- 7) " 
        elif num<=170:
            return str(num)+" (+/- 12) " 
        elif num<=180:
            return str(num)+" (+/- 14) "
        else:
            return str(num)+" (+/- 13) " 
        
    def predict_rank_u(self,num):
        if num<0:
            return "1 (+/- 2)"
        if num<10:
            return str(num)+" (+/- 2)"
        if num<30:
            return str(num)+" (+/- 2)"
        if num<40:
            return str(num)+" (+/- 4)"
        if num<50:
            return str(num)+" (+/- 6)"
        if num<60:
            return str(num)+" (+/- 7)"
        if num<70:
            return str(num)+" (+/- 4)"
        if num<80:
            return str(num)+" (+/- 2)"
        if num<90:
            return str(num)+" (+/- 2)"
        else:
            return str(num)+" (+/- 2)"
        
    def predict_rank_o(self,num):
        if num<0:
            return "1 +/- 2"
        if num<10:
            return str(num)+" (+/- 2)"
        if num<30:
            return str(num)+" (+/- 2)"
        if num<40:
            return str(num)+" (+/- 4)"
        if num<50:
            return str(num)+" (+/- 1)"
        if num<60:
            return str(num)+" (+/- 2)"
        if num<70:
            return str(num)+" (+/- 2)"
        if num<80:
            return str(num)+" (+/- 2)"
        if num<90:
            return str(num)+" (+/- 1)"
        else:
            return str(num)+" (+/- 7)"        

rank_uni_para=[]
rpc_uni_para=[]
tlr_uni_para=[]
go_uni_para=[]
oi_uni_para=[]
ppn_uni_para=[]


rank_over_para=[]
rpc_over_para=[]
tlr_over_para=[]
go_over_para=[]
oi_over_para=[]
ppn_over_para=[]


rank_engg_para=[]
rpc_engg_para=[]
tlr_engg_para=[]
go_engg_para=[]
oi_engg_para=[]
ppn_engg_para=[]


def uni_clear():
    rank_uni_para.clear()
    rpc_uni_para.clear()
    tlr_uni_para.clear()
    go_uni_para.clear()
    oi_uni_para.clear()
    ppn_uni_para.clear()

    
    
def over_clear():
    rank_over_para.clear()
    rpc_over_para.clear()
    tlr_over_para.clear()
    go_over_para.clear()
    oi_over_para.clear()
    ppn_over_para.clear()

def engg_clear():
    rank_engg_para.clear()
    rpc_engg_para.clear()
    tlr_engg_para.clear()
    go_engg_para.clear()
    oi_engg_para.clear()
    ppn_engg_para.clear()    


app = Flask(__name__)
app.secret_key = 'secret!!'







@app.route('/')
def home():
	return render_template('index.html')

@app.route('/engg/')
def engg():
    return render_template('engg.html')

@app.route('/university/')
def university():
    return render_template('university.html')

@app.route('/overall/')
def overall():
    return render_template('overall.html')

@app.route('/index')
def index():
    engg_clear()
    over_clear()
    uni_clear()
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        uni_clear()
        over_clear()
        tlr = float(request.form['tlr'])
        rpc = float(request.form['rpc'])
        go = float(request.form['go'])
        oi =float(request.form['oi'])
        perception = float(request.form['ppn'])

        score=(tlr*0.3)+(rpc*0.3)+(go*0.2)+(oi*0.1)+(perception*0.1)
        data = np.array([score]).reshape(1,-1)
        pre_prediction=model1.fit_transform(data)
        my_prediction = model.predict(pre_prediction)
        
        
        tlr_engg_para.append(float(tlr))
        rpc_engg_para.append(float(rpc))
        go_engg_para.append(float(go))
        oi_engg_para.append(float(oi))
        ppn_engg_para.append(float(perception))
               
        df_engg = pd.DataFrame({"TLR":tlr_engg_para,"RPC":rpc_engg_para,"GO":go_engg_para,"OI":oi_engg_para,"PPN":ppn_engg_para})
        df_engg.index = np.arange(df_engg.shape[0])+1
        
        
        x=my_prediction[0]
        range1=modify1()
        ranges=range1.find_range(int(x))
        if x<=200:
            rank_engg_para.append(ranges)
            df_engg['RANK']=rank_engg_para
            return render_template('engg.html', prediction="The predicted rank might be in range : " + ranges,tables=[df_engg.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])
        else:
            rank_engg_para.append(">150")
            df_engg['RANK']=rank_engg_para            
            return render_template('engg.html',prediction="The rank for this score is greater than 200",tables=[df_engg.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])
        ##return render_template('result.html', prediction=data.shape)


@app.route('/predictUni', methods=['POST'])
def predictUni():
    if request.method == 'POST':
        over_clear()
        engg_clear()
        tlr_u = float(request.form['tlr_u'])
        rpc_u = float(request.form['rpc_u'])
        go_u = float(request.form['go_u'])
        oi_u =float(request.form['oi_u'])
        perception_u = float(request.form['perception_u'])
        score_u=(tlr_u*0.3)+(rpc_u*0.3)+(go_u*0.2)+(oi_u*0.1)+(perception_u*0.1)
        data_u = np.array([score_u]).reshape(1,-1)
        pre_prediction_u=model_uni.fit_transform(data_u)
        my_prediction_u=model_uni1.predict(pre_prediction_u)
        x1=my_prediction_u[0,0]
        
        range2=modify1()
        ranges2=range2.predict_rank_u(int(x1))

                  
        tlr_uni_para.append(float(tlr_u))
        rpc_uni_para.append(float(rpc_u))
        go_uni_para.append(float(go_u))
        oi_uni_para.append(float(oi_u))
        ppn_uni_para.append(float(perception_u))

        df = pd.DataFrame({"TLR":tlr_uni_para,"RPC":rpc_uni_para,"GO":go_uni_para,"OI":oi_uni_para,"PPN":ppn_uni_para})
        df.index = np.arange(df.shape[0])+1
        
        if x1<=140:
            rank_uni_para.append(ranges2)
            df['RANK']=rank_uni_para
            return render_template('university.html', prediction="The predicted rank might be in range : " + ranges2,tables=[df.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])
        else:
            rank_uni_para.append(">150")
            df['RANK']=rank_uni_para            
            return render_template('university.html',prediction="The rank for this score is greater than 150",tables=[df.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])            
        



@app.route('/predictOver', methods=['POST'])
def predictOver():
    if request.method == 'POST':
        
        uni_clear()
        engg_clear()
        
        
        
        tlr_o = float(request.form['tlr_o'])
        rpc_o = float(request.form['rpc_o'])
        go_o = float(request.form['go_o'])
        oi_o =float(request.form['oi_o'])
        perception_o = float(request.form['perception_o'])
        score_o=(tlr_o*0.3)+(rpc_o*0.3)+(go_o*0.2)+(oi_o*0.1)+(perception_o*0.1)
        data_o = np.array([score_o]).reshape(1,-1)
        pre_prediction_o=model_over.fit_transform(data_o)
        my_prediction_o=model_over1.predict(pre_prediction_o)
        x2=my_prediction_o[0,0]
        
        range3=modify1()
        ranges3=range3.predict_rank_o(int(x2))        
        
        tlr_over_para.append(float(tlr_o))
        rpc_over_para.append(float(rpc_o))
        go_over_para.append(float(go_o))
        oi_over_para.append(float(oi_o))
        ppn_over_para.append(float(perception_o))
                
        df_over = pd.DataFrame({"TLR":tlr_over_para,"RPC":rpc_over_para,"GO":go_over_para,"OI":oi_over_para,"PPN":ppn_over_para})
        df_over.index = np.arange(df_over.shape[0])+1

        if x2<=140:
            rank_over_para.append(ranges3)
            df_over['RANK']=rank_over_para
            return render_template('overall.html', prediction="The predicted rank might be in range : " + ranges3,tables=[df_over.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])
        else:
            rank_over_para.append(">150")
            df_over['RANK']=rank_over_para            
            return render_template('overall.html',prediction="The rank for this score is greater than 150",tables=[df_over.to_html(classes='data')],titles=['.','TLR','RPC','GO','OI','PPN','RANK'])

if __name__ == '__main__':
	app.run(debug=True)
