from flask import Flask
from .routes import ui

def create_app():
    app = Flask(__name__)
    app.secret_key = "replace-me-in-prod"
    app.register_blueprint(ui)
    return app
