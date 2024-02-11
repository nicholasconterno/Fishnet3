from flask import Flask
import os

def create_app():
    app = Flask(__name__)

    from .routes import app_routes, IMG_FOLDER

    app.register_blueprint(app_routes)   
    # Clear out the IMG_FOLDER directory
    for file in os.listdir(IMG_FOLDER):
        os.remove(os.path.join(IMG_FOLDER, file))
    return app
