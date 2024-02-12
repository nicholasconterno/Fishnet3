from flask import Blueprint, request, render_template, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
import os
from .utils import object_detect

app_routes = Blueprint('app_routes', __name__)

# Folder to save the uploaded images temporarily
IMG_FOLDER = os.path.join(os.getcwd(), 'app/temp_images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    '''
    Check if the file extension is allowed.
    '''
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app_routes.route('/')
def index():
    '''
    Endpoint to render the main page.
    '''
    return render_template('upload.html')

@app_routes.route('/upload', methods=['POST'])
def upload_file():
    '''
    Endpoint to upload a file and process it with object detection.
    '''
    if 'file' not in request.files:
        return render_template('upload.html', message='No file part. Please try again.')

    file = request.files['file']

    if file.filename == '':
        return render_template('upload.html', message='No selected file. Please try again.')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(IMG_FOLDER, filename)
        print(f"Saving file to {filepath}")
        file.save(filepath)

        # Process the image with object detection and keep new filename
        rcnn_img_name, fishnet_img_name, rf_img_name = object_detect(filepath)
        print(f"Processed images: {rcnn_img_name}, {fishnet_img_name}, {rf_img_name}")
        if rcnn_img_name and fishnet_img_name and rf_img_name:
            return redirect(url_for('app_routes.display_image', filenames=[rcnn_img_name, fishnet_img_name, rf_img_name]))
        else:
            return render_template('upload.html', message='Something went wrong. Please try again.')
    else:
        return render_template('upload.html', message='File type not allowed. Please try again with a valid image file.')

@app_routes.route('/display/<filename>')
def display_images(filenames):
    '''
    Endpoint to display the processed image.
    '''
    return render_template('display_images.html', filenames=filenames)

@app_routes.route('/processed/<filename>')
def serve_processed_image(filename):
    '''
    Endpoint to serve the processed image.
    '''
    return send_from_directory(IMG_FOLDER, filename)

