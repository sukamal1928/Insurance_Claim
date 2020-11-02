from flask import Flask,render_template,request
import pickle
import numpy as np

file = open("model.ser","rb")
dt = pickle.load(file)
file.close()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_claim", methods=["POST"])
def predict_claim():
    age = request.form["age"]
    sex = request.form["sex"]
    bmi = request.form["bmi"]
    children = request.form["children"]
    smoker = request.form["smoker"]
    region = request.form["region"]
    charges = request.form["charges"]
    
    output = dt.predict(np.array([[age,sex,bmi,children,smoker,region,charges]]))
    
    return str(output[0])
    


if __name__ == '__main__':
   app.run(port=5001)