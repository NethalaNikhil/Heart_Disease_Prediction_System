from flask import Flask, render_template,url_for,request,redirect
import pickle
import numpy as np
#import sklearn
#print(sklearn.__version__)
#pip install --upgrade scikit-learn==1.2.2

model=pickle.load(open("E:\Heart_Disease_Prediction_System\RFC.pkl",'rb'))
app = Flask(__name__)
@app.route('/',methods=["GET","POST"])
def home():
    return render_template("homepage.html")
@app.route('/input')
def input():
    return render_template('input.html')
@app.route('/home1',methods=['POST'])
def home1():
    data1=request.form['a']
    data2=request.form['b']
    data3=request.form['c']
    data4=request.form['d']
    data5=request.form['e']
    data6=request.form['f']
    data7=request.form['g']
    data8=request.form['h']
    data9=request.form['i']
    data10=request.form['j']
    data11=request.form['k']


    data1 = np.asarray(data1, dtype='int32')
    data2 = np.asarray(data2, dtype='int64')
    data3 = np.asarray(data3, dtype='int32')
    data4 = np.asarray(data4, dtype='int32')
    data5 = np.asarray(data5, dtype='int32')
    data6 = np.asarray(data6, dtype='int32')
    data7 = np.asarray(data7, dtype='int64')
    data8 = np.asarray(data8, dtype='int64')
    data9 = np.asarray(data9, dtype='int64')
    data10 = np.asarray(data10, dtype='int64')
    data11 = np.asarray(data11, dtype='int64')

    databmi=data3/data4
    arr=np.array([[data1,data2,data5,data6,data7,data8,data9,data10,data11,databmi]])
    pred=model.predict(arr)
    return render_template("after.html",data=pred[0])
if __name__=="__main__":
    app.run(debug=True)
