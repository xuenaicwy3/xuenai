from flask import Flask

import settings
from apps.article.view import article_bp
from apps.goods.view import goods_bp
from apps.user.view import user_bp
from apps.user.views import user_bp1
from exts import db,bootstrap



def create_app():
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config.from_object(settings.DevelopmentConfig)
    # 初始化配置db
    db.init_app(app=app)  # 将db对象与app进行了关联
    # 初始化bootstrap
    bootstrap.init_app(app=app)
    # 注册蓝图
    #app.register_blueprint(user_bp)
    app.register_blueprint(user_bp1)
    app.register_blueprint(article_bp)
    #app.register_blueprint(goods_bp)
    #print(app.url_map)
    return app