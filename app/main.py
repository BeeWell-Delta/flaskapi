from fileinput import filename
from flask import Flask, jsonify, request
from torch_utils import transform_image
# get_prediction

import io

from PIL import Image

from torch_utils import get_prediction

import secrets as sc  
import string  


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/get_data', methods=['GET'])
def reply_smth():
    if request.method == 'GET':
        with open('data.txt') as f:
            first_line = f.readline()
            if first_line != "":
                values = first_line.split(';')
                sec = str(values[1])
                sec = sec[0:-1]
                data = {'prediction' : str(values[0]), 'img_path' : sec}
            else:
                data = {'val' : 'val'}
            return jsonify(data)

@app.route('/predict', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            # data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
             
            image_path = "static"

            sequence = string.ascii_letters + string.digits  
            while True:  
                img_name = ''.join(sc.choice(sequence) for i in range(8))  
                if (any(c.islower() for c in img_name) and any(c.isupper()  
                for c in img_name) and sum(c.isdigit() for c in img_name) >= 3):  
                    break

            img = Image.open(io.BytesIO(img_bytes))

            img.save(f"{image_path}/{img_name}.png")

            img_path = f"http://192.168.75.83:5000/static/{img_name}.png"

            data =  {'prediction': prediction, 'img_path' : img_path}
            write_data = prediction + ";" + img_path + "\n"

            with open("data.txt", 'r+') as file:
                readcontent = file.read()   
                file.seek(0, 0)
                file.write(write_data)
                file.write(readcontent)  

            file.close()


            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})
    #LOAD THE IMAGE
    #IMAGE-> TENSOR
    #predict
    #return data
    return jsonify({'result' : 1 })