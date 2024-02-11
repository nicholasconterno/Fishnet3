from flask import Blueprint, request, render_template, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
import os
from .utils import process_image

app_routes = Blueprint('app_routes', __name__)

# Configure your Flask application to save uploaded files
IMG_FOLDER = os.path.join(os.getcwd(), 'app/temp_images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app_routes.route('/')
def index():
    return render_template('upload.html')

@app_routes.route('/upload', methods=['POST'])
def upload_file():
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
        processed_image_path, processed_image_name = process_image(filepath, IMG_FOLDER)
        if processed_image_path:
            processed_filename = os.path.basename(processed_image_path)
            return redirect(url_for('app_routes.display_image', filename=processed_filename))
        else:
            return render_template('upload.html', message='Something went wrong. Please try again.')
    else:
        return render_template('upload.html', message='File type not allowed. Please try again with a valid image file.')

@app_routes.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(IMG_FOLDER, filename)

