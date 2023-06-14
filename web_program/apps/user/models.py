# ORM   类 ----》 表
# 类对象  ---〉表中的一条记录
from datetime import datetime

from exts import db



# create table user(id int primarykey auto_increment,username varchar(20) not null,..)
class DoctorInfo(db.Model):
    # db.Column(类型，约束)  映射表中的列
    #
    '''
    类型： 
    db.Integer      int
    db.String(15)   varchar(15)
    db.Datetime     datetime
    
    '''
    tableName = 'doctor_info'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    doctor_name = db.Column(db.String(10), nullable=False)
    doctor_level = db.Column(db.String(12), nullable=False)
    doctor_kind = db.Column(db.String(11), unique=True)
    doctor_Belonging = db.Column(db.String(12), nullable=False)
    doctor_score = db.Column(db.String(11), unique=True)
    doctor_inquiry = db.Column(db.String(12), nullable=False)
    doctor_goodFor = db.Column(db.String(255), nullable=False)
    pic_see_price = db.Column(db.String(255), nullable=False)
    video_see_price = db.Column(db.String(255), nullable=False)
    doctor_detail = db.Column(db.String(255), nullable=False)
    # isdelete = db.Column(db.Boolean, default=False)
    # email = db.Column(db.String(20))
    rdatetime = db.Column(db.DateTime, default=datetime.now)


    def __str__(self):
        return self.doctor_name


# class UserInfo(db.Model):
#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     realname = db.Column(db.String(20))
#     gender = db.Column(db.Boolean, default=False)
#
#     def __str__(self):
#         return self.realname

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(11), unique=True, nullable=False)
    email = db.Column(db.String(30))
    icon = db.Column(db.String(100))
    isdelete = db.Column(db.Boolean, default=False)
    rdatetime = db.Column(db.DateTime, default=datetime.now)
    # 增加一个字段
    articles = db.relationship('Article', backref='user')
    comments = db.relationship('Comment', backref='user')

    def __str__(self):
        return self.username


class Photo(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    photo_name = db.Column(db.String(100), nullable=False)
    photo_datetime = db.Column(db.DateTime, default=datetime.now)
    user_id = db.Column(db.String(20), db.ForeignKey('user.id'), nullable=False)
    photo_predict = db.Column(db.String(1000), nullable=False)
    photo_advise = db.Column(db.String(1000), nullable=False)
    def __str__(self):
        return self.photo_name

