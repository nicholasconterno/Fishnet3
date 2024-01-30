from flask import Blueprint, request, render_template, redirect, url_for, session
from .utils import process_image

app_routes = Blueprint('app_routes', __name__)


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

    if file:
        if process_image(file):
            return render_template('upload.html', message='Thank you for contributing')
        else:
            return render_template('upload.html', message='Something went wrong. Please try again.')