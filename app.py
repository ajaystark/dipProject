from flask import Flask,render_template,request,json,Response,jsonify,send_from_directory,send_file
import json
from datetime import datetime
import os
from werkzeug.utils import secure_filename
from edge import get_density

uploads_dir='uploads'
app = Flask(__name__,template_folder="templates",static_url_path='/static',static_folder='static')

@app.errorhandler(404)
def not_found(error=None):
    message = {
        'status':404,
        'message':'Not Found: ' + request.url,
    }
    resp = jsonify(message)
    resp.status_code = 404
    return resp

@app.route('/',methods=['GET','POST'])
def encrypt_page():
    if request.method=='GET':
        return render_template('encrypt.html')
    if request.method=='POST':
        file=request.files['file']
        # key=request.form['key']
        
        file_path=os.path.join(uploads_dir, secure_filename(file.filename))
        file.save(os.path.join(file_path))

        result=get_density(file_path)

        return render_template('encrypt.html',result=result)

if __name__=='__main__':
    app.run(debug=True)