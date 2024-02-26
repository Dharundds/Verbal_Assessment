from flask import Flask,request,jsonify
from flask_cors import CORS
from speech_diarisation import main as speech_diazisation
import json
import os

app = Flask(__name__)
CORS(app,resources={r"/*":{"origins": "http://localhost:3000"}})
PORT = 5000

@app.route("/",methods=["POST"])
def home():
    print(request.files)
    if 'audioFile' not in request.files:
        return jsonify({"status":"Error",'message':"No files"})
    
    file = request.files['audioFile']
    if file.filename == "":
        return jsonify({"status":"Error","message":"No Selected File"})

    print("File = ",file)
    filename = './media/'+file.filename
    file.save(filename)
    return jsonify({"status":201,"message":"File Uploaded Successfully"})


@app.route("/getResult",methods=["POST"])
def getResult():
    try:
        filename = "./media/"+request.json.get("filename")
        print("filename = ", filename)
        spkr_dict = {}
        if filename!=None:
            if os.path.exists(filename):
                spkr_dict = speech_diazisation(filename)
            else:
                return jsonify({"status":404,"message":f"File {filename} Not Found","data":"null"})
        # session.clear()
        return jsonify({"status":200,"message":"Retrieve information successfully","data":spkr_dict})
    except Exception as e:
        return jsonify({"status":500,"message":"Some error occured","data":jsonify({"error":str(e)})})

    

if __name__ == "__main__":
    app.run(port=5000,debug=True)