from flask import Flask, request, render_template, redirect, url_for,flash
import sqlite3
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/UserHome')
def UserHome():
    return render_template('user/userhome.html')

# Route to add a new record (INSERT) student data to the database
@app.route("/addrec", methods = ['POST', 'GET'])
def addrec():
    # Data will be available from POST submitted by the form
    if request.method == 'POST':
        try:
            nm = request.form['nm']
            addr = request.form['add']
            city = request.form['city']
            zip = request.form['zip']
            loginid = request.form['loginid']
            email = request.form['email']
            password = request.form['password']
            # Connect to SQLite3 database and execute the INSERT
            with sqlite3.connect('database.db') as con:
                cur = con.cursor()
                cur.execute("INSERT INTO students (name, loginid, email,  password, addr, city, zip) VALUES (?,?,?,?,?,?,?)",(nm, loginid, email, password, addr, city, zip))
                con.commit()
                msg = "Record successfully added to database"
        except:
            con.rollback()
            msg = "Error in the INSERT"
        finally:
            con.close()
            # Send the transaction message to result.html
            return render_template('result.html',msg=msg)       
        
        
@app.route("/enternew")
def enternew():
    return render_template("student.html")


@app.route('/list')
def list():
    # Connect to the SQLite3 datatabase and 
    # SELECT rowid and all Rows from the students table.
    con = sqlite3.connect("database.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT rowid, * FROM students")

    rows = cur.fetchall()
    con.close()
    # Send the results of the SELECT to the list.html page
    return render_template("list.html",rows=rows)



@app.route("/edit", methods=['POST','GET'])
def edit():
    if request.method == 'POST':
        try:
            # Use the hidden input value of id from the form to get the rowid
            id = request.form['id']
            # Connect to the database and SELECT a specific rowid
            con = sqlite3.connect("database.db")
            con.row_factory = sqlite3.Row

            cur = con.cursor()
            cur.execute("SELECT rowid, * FROM students WHERE rowid = " + id)

            rows = cur.fetchall()
        except:
            id=None
        finally:
            con.close()
            # Send the specific record of data to edit.html
            return render_template("edit.html",rows=rows)
        
        


# Route used to execute the UPDATE statement on a specific record in the database
@app.route("/editrec", methods=['POST'])
def editrec():
    if request.method == 'POST':
        con = None
        try:
            print(request.form)
            rowid = request.form['rowid']
            nm = request.form['nm']
            addr = request.form['add']
            city = request.form['city']
            zip = request.form['zip']
            loginid = request.form['loginid']
            email = request.form['email']
            password = request.form['password']

            with sqlite3.connect('database.db') as con:
                cur = con.cursor()
                cur.execute("""
                    UPDATE students SET name=?, addr=?, city=?, zip=?, loginid=?, email=?, password=?
                    WHERE rowid=?
                """, (nm, addr, city, zip, loginid, email, password, rowid))

                con.commit()
                msg = "Record successfully edited in the database"
        except Exception as e:
            if con:
                con.rollback()
            msg = f"Error in the Edit: {str(e)}"
        finally:
            if con:
                con.close()
            return render_template('result.html', msg=msg)

       
@app.route("/delete", methods=['POST', 'GET'])
def delete():
    if request.method == 'POST':
        try:
            # Use the hidden input value of id from the form to get the rowid
            rowid = request.form['id']
            # Connect to the database and DELETE a specific record based on rowid
            with sqlite3.connect('database.db') as con:
                cur = con.cursor()
                cur.execute("DELETE FROM students WHERE rowid = ?", (rowid,))
                con.commit()
                msg = "Record successfully deleted from the database"
        except:
            con.rollback()
            msg = "Error in the DELETE"
        finally:
            con.close()
            # Send the transaction message to result.html
            return render_template('result.html', msg=msg)

        
        
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()
        loginid = request.form['loginid']
        password = request.form['password']
        print(loginid, password)
        query = "SELECT loginid, password FROM students WHERE loginid=? AND password=?"
        cursor.execute(query, (loginid, password))
        results = cursor.fetchall()        
        if not results:
            error_message = 'Invalid login credentials. Please try again.'
            return render_template('login.html', error=error_message)
        else:
            return render_template('user/userhome.html')
    return render_template('login.html')   

'''
here ml code start here 
'''
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from PIL import Image
from keras.models import load_model

app.config['UPLOAD_FOLDER'] = 'media'
app.config['MODEL_FOLDER'] = 'static/model'

# Load Dataset
skin_df = pd.read_csv('HAM10000_metadata.csv')
data_folder_name = "static/HAM10000_images1"
ext = ".jpg"

# Assuming skin_df is your DataFrame containing image IDs in the column 'image_id'
skin_df["path"] = [data_folder_name + "/" + img_id + ext for img_id in skin_df["image_id"]]

# Check and process only existing files
existing_files_mask = skin_df["path"].apply(lambda x: os.path.exists(x))
skin_df = skin_df[existing_files_mask]  # Filtering out rows with non-existing files

# Process images
skin_df["image"] = skin_df["path"].map(lambda x: np.asarray(Image.open(x).resize((100, 75))))
skin_df["dx_idx"] = pd.Categorical(skin_df["dx"]).codes

# Standardization - Normalization
x_train = np.asarray(skin_df["image"].tolist())
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_train = (x_train - x_train_mean) / x_train_std

# One-Hot Encoding
num_classes = skin_df["dx"].nunique()

# Load Models
model_1 = load_model(os.path.join(app.config['MODEL_FOLDER'], 'my_model_1.h5'))
model_2 = load_model(os.path.join(app.config['MODEL_FOLDER'], 'my_model_2.h5'))

def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((100, 75))
    img = np.asarray(img).reshape(1, 75, 100, 3)
    img = (img - x_train_mean) / x_train_std
    return img

def classify_image(img, model):
    prediction = model.predict(img)[0]
    predicted_class = np.argmax(prediction)
    predicted_cancer = skin_df.dx.unique()[predicted_class]
    return prediction, predicted_cancer

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            
            model_name = request.form.get('model')
            if model_name == 'Model_1':
                model = model_1
            else:
                model = model_2
            
            img = preprocess_image(img_path)
            prediction, predicted_cancer = classify_image(img, model)
            
            prediction_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pred_' + filename)
            Image.open(img_path).save(prediction_img_path)
            print(f"Prediction image saved to: {prediction_img_path}")

            class_names = skin_df.dx.unique()
            print(class_names)
            # enumerated_predictions = list(enumerate(prediction))
            enumerated_predictions = enumerate(prediction)
            print(enumerated_predictions)
            return render_template('user/result.html', img_path=prediction_img_path, enumerated_predictions=enumerated_predictions, predicted_cancer=predicted_cancer, class_names=class_names)
        else:
            flash('Invalid file format')
            return redirect(request.url)
    return render_template('user/upload.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}  
    
    

    


'''
end the ml code 
'''



if __name__ == '__main__':
    app.run(debug=True,port=8000)
