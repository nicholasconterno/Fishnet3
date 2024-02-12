from flask import Flask
import os

IMG_FOLDER = os.path.join(os.getcwd(), 'app/temp_images')

def create_app():
    '''
    Initialize the Flask application.
    '''
    app = Flask(__name__)

    from .routes import app_routes

    app.register_blueprint(app_routes)
    # Check if the IMG_FOLDER directory exists and create it if not
    if not os.path.exists(IMG_FOLDER):
        os.makedirs(IMG_FOLDER)   
    # Clear out the IMG_FOLDER directory
    for file in os.listdir(IMG_FOLDER):
        os.remove(os.path.join(IMG_FOLDER, file))
    return app
