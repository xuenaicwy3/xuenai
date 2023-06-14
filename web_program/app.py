from flask import Flask
from apps import create_app
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager
from apps.user.models import DoctorInfo, User, Photo
from apps.article.models import *
from apps.goods.models import *
from exts import db

# app = Flask(__name__)y

app = create_app()
manager = Manager(app=app)

# 命令工具
migrate = Migrate(app=app, db=db)
manager.add_command('db', MigrateCommand)

@manager.command
def init():
    print('初始化')

# @app.route('/')
# def index():
#     return "hello world"

if __name__ == '__main__':
    manager.run()