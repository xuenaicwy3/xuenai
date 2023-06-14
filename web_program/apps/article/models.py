from datetime import datetime

from exts import db
'''
说明今天最后的报错原因：
Article表中已经有3条数据了，我在Article模型中有增加了一列，
此列是：type_id = db.Column(db.Integer, db.ForeignKey('type.id'), nullable=False)
注意我在里面添加了nullable=False,就是说明不能为空。

问题就是我在已经存在数据的表中添加了一列而且还要求此列数据不能为空，这个是冲突的！!!
故产生的错误。

解决办法：
nullable=False不添加，就是允许为空，
或者添加一个默认值也可以。

'''

class Article_type(db.Model):
    __tablename__ = 'type'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    type_name = db.Column(db.String(20), nullable=False)
    articles = db.relationship('Article', backref='articletype')


class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(50), nullable=False)
    content = db.Column(db.Text, nullable=False)
    pdatetime = db.Column(db.DateTime, default=datetime.now)
    click_num = db.Column(db.Integer, default=0)
    save_num = db.Column(db.Integer, default=0)
    love_num = db.Column(db.Integer, default=0)
    # 外键 同步到数据库的外键关系
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    type_id = db.Column(db.Integer, db.ForeignKey('type.id'), nullable=False)
    comments = db.relationship('Comment', backref='article')


class Comment(db.Model):
    # 自定义表的名字
    __tablename__ = 'comment'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    comment = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    article_id = db.Column(db.Integer, db.ForeignKey('article.id'))
    cdatetime = db.Column(db.DateTime, default=datetime.now)

    def __str__(self):
        return self.comment
