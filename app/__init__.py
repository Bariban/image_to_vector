from flask import Flask
from .routes import app

def create_app():
    app.config.from_object('instance.config')
    return app