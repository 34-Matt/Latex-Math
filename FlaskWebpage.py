from flask import Flask, jsonify, request, render_template
import numpy as np
import joblib
import cv2
import traceback

from Box_Character import Box_Character
from equationClass import equation
from TrainCNN import loadModel

app = Flask(__name__)
model = None


@app.route('/')
def main():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('index.html')

@app.route('/run',methods=["POST"])
def run():
    global model
    try:
        # Initialize equation storage
        LatexEq = equation([],[])
        
        # Grab user image
        image = request.files['file'].read()
        arr = cv2.imdecode(np.fromstring(image,np.uint8), cv2.IMREAD_UNCHANGED)
        
        # Need to breakup images into parts
        images = Box_Character(arr)
        
        # Predict each part and append to equation
        for im in images:
            im = im.reshape((1,45,45,1))
            pred = model.predict(im).argmax()
            LatexEq.appendTerm(pred,0)
            
        # Latex format
        latex = LatexEq.printLatex()
        
        # Send to webpage
        return jsonify({
            "message": f"Latex Format: {latex}",
            "latex":latex
        })
    
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "message" : f"An error occurred. {e}"
        })

@app.route('/run-ui')
def run_ui():
    return render_template("process.html")

if __name__ == '__main__':
    model = loadModel((45,45,1),66,'training/cp-0396.ckpt.index')
    app.run(debug=True)