
from flask import Flask, render_template, request
import os
 
PORT = 3000

app = Flask (__name__ ,template_folder="temp")

@app.route('/', methods=['GET', 'POST'])
def uploadfiles():
    if request.method == 'POST':
        file = request.files("audio")  #html label name
        if file  :
            filename = 'audio.mp3'  
            file.save(os.path.join('uploads',filename))  #saving the file in uploads directory
            return "file uploaded"
    return render_template("index.html") # homepage

if __name__ == '__main__':
    if not os.path.exists('uploads'):  # if directory does not exits - new directory created
        os.makedirs('uploads')
    app.run(port=PORT,debug=True)


    
