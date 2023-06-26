import pandas as pd
import flask
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
import pickle

app=Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def ValuePredictor(data):
    model = pickle.load(open('checkpoints\model.pkl', 'rb'))
    print(data)
    prediction = model.predict(data)

    if prediction==1:
        result='Se gana'
    else:
        result='Se pierde'

    return result

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        params = request.form.to_dict()
        params = list(params.values())
        
        params = list(map(float, params))
        d = {
            'Team': [params[0]], 
            'RoundStartingEquipmentValue': [params[1]], 
            'TeamStartingEquipmentValue': [params[2]]
        }
        d = pd.DataFrame(data=d)
        sc = MinMaxScaler()
        d = sc.fit_transform(d)
        result = ValuePredictor(d)

        return flask.render_template("result.html", result=result)

if __name__=="__main__":
    app.run(port=5001)