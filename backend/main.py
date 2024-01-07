from flask import Flask,request,jsonify
from flask_cors import CORS

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
    file.save('./'+file.filename)

    return jsonify({"status":"Success","message":"File Uploaded Successfully"})

if __name__ == "__main__":
    app.run(port=5000,debug=True)