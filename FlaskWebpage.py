from flask import Flask, jsonify, request, render_template
import numpy as np
import joblib
import cv2
import traceback

#from TrainCNN import loadLatestModel

app = Flask(__name__)
model = None


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/run',methods=["POST"])
def run():
    global model
    try:
        image = request.files['file'].read()
        arr = cv2.imdecode(np.fromstring(image,np.uint8), cv2.IMREAD_UNCHANGED)
        my_image = arr / 255.0
        
        # Need to breakup images into parts
        
        #my_image = my_image.reshape(1,45,45,1)
        #pred = model.predict(my_image)
            
        # Need to add prediction to latex format
        latex = "$Hi Beech$"
            
        # Latex format
        
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
    #model = loadLastestModel((45,45,1),66)
    app.run(debug=True)