1.查询：
查询所有： 模型类.query.all()    ~  select * from user;
如果有条件的查询：
         模型类.query.filter_by(字段名 = 值)   ～  select * from user where 字段=值；
         模型类.query.filter_by(字段名 = 值).first()   ～  select * from user where 字段=值 limit..；

select * from user where age>17 and gender='男'；
select * from user where username like 'zhang%';
select * from user where rdatetime> xxx and rdatetime < xxx;

         模型类.query.filter()  里面是布尔的条件   模型类.query.filter(模型名.字段名 == 值)
         模型类.query.filter_by()  里面是一个等值   模型类.query.filter_by(字段名 = 值)


***** 模型类.query.filter() ******
1. 模型类.query.filter().all()   -----> 列表
2. 模型类.query.filter().first()  ----->对象
3.User.query.filter(User.username.endswith('z')).all()   select * from user where username like '%z';
  User.query.filter(User.username.startswith('z')).all()  # select * from user where username like 'z%';
  User.query.filter(User.username.contains('z')).all()  # select * from user where username like '%z%';
  User.query.filter(User.username.like('z%')).all()

  多条件：
  from sqlalchemy import or_, and_,not_
  并且： and_    获取： or_   非： not_
  User.query.filter(or_(User.username.like('z%'), User.username.contains('i'))).all()
   类似： select * from user where username like 'z%' or username like '%i%';

  User.query.filter(and_(User.username.contains('i'), User.rdatetime.__gt__('2020-05-25 10:30:00'))).all()
   # select * from user where username like '%i%' and rdatetime < 'xxxx'

  补充：__gt__,__lt__,__ge__(gt equal),__le__ （le equal）  ----》通常应用在范围（整型，日期）
       也可以直接使用 >  <  >=  <=  !=

  User.query.filter(not_(User.username.contains('i'))).all()

  18 19 20 17 21 22 ....
  select * from user where age in [17,18,20,22];


排序：order_by

    user_list = User.query.filter(User.username.contains('z')).order_by(-User.rdatetime).all()  # 先筛选再排序
    user_list = User.query.order_by(-User.id).all()  对所有的进行排序
    注意：order_by(参数)：
    1。 直接是字符串： '字段名'  但是不能倒序
    2。 填字段名： 模型.字段    order_by(-模型.字段)  倒序

限制： limit
    # limit的使用 + offset
    # user_list = User.query.limit(2).all()   默认获取前两条
    user_list = User.query.offset(2).limit(2).all()   跳过2条记录再获取两条记录


 总结：
 1. User.query.all()  所有
 2. User.query.get(pk)  一个
 3. User.query.filter()   *   ？？？？？？？
     如果要检索的字段是字符串（varchar，db.String）:
       User.username.startswith('')
       User.username.endswith('')
       User.username.contains('')
       User.username.like('')
       User.username.in_(['','',''])
       User.username == 'zzz'
    如果要检索的字段是整型或者日期类型：
       User.age.__lt__(18)
       User.rdatetime.__gt__('.....')
       User.age.__le__(18)
       User.age.__ge__(18)
       User.age.between(15,30)

     多个条件一起检索： and_, or_
     非的条件： not_

     排序：order_by()
     获取指定数量： limit() offset()
 4. User.query.filter_by()


 删除:
 两种删除：
 1。逻辑删除（定义数据库中的表的时候，添加一个字段isdelete，通过此字段控制是否删除）
 id = request.args.get(id)
 user = User.query.get(id)
 user.isdelete = True
 db.session.commit()

 2。物理删除(彻底从数据库中删掉)
 id = request.args.get(id)
 user = User.query.get(id)
 db.session.delete(user)
 db.session.commit()


 更新:
 id = request.args.get(id)
 user = User.query.get(id)
 # 修改对象的属性
 user.username= xxxx
 user.phone =xxxx
 # 提交更改
 db.session.commit()


 两张表


 主要理解多张数据库表的关系。

明确：一个项目肯定会有多张表，确定表与表之间的关系最重要。
在开始项目前必须要确定表与表的关系

单独一张表： User 是不行的。user要与其他的建立联系。

以student和班级clazz为例：
一个班级是有多名学生的

----模板
     --html
     --js
     --css
     --images
--- 5.27 ---
使用flask-bootstrap:
步骤：
1。pip install flask-bootstrap
2.进行配置：
 from flask-bootstrap import Bootstrap
 bootstrap = Bootstrap()

 在__init__.py中进行初始化：
 # 初始化bootstrap
 bootstrap.init_app(app=app)
3。内置的block：
{% block title %}首页{% endblock %}

{% block navbar %} {% endblock %}

{% block content %} {% endblock %}

{% block styles %} {% endblock %}

{% block srcipts %} {% endblock %}
{% block head %} {% endblock %}

{% block body %} {% endblock %}


flask-bootstrap
bootstrap-flask  -----> 卸载

密码加密：
注册：
generate_password_hash(password)  ----> 加密后的密码
sha256加密$salt$48783748uhr8738478473...

登录：
check_password_hash(pwdHash,password)  -----> bool:False,True

会话机制：
1。cookie方式：

  保存：
    通过response对象保存。
    response = redirect(xxx)
    response = render_template(xxx)
    response = Response()
    response = make_response()
    response = jsonify()
    # 通过对象调用方法
    response.set_cookie(key,value,max_age)
    其中max_age表示过期时间，单位是秒
    也可以使用expires设置过期时间，expires=datetime.now()+timedelta(hour=1)

  获取：
    通过request对象获取。
    request.form.get()
    request.args.get()
    cookie也在request对象中
    request.cookies.get(key) ----> value

  删除：
     通过response对象删除。 把浏览器中的key=value删除了
    response = redirect(xxx)
    response = render_template(xxx)
    response = Response()
    response = make_response()
    response = jsonify()
    # 通过对象调用方法
    response.delete_cookie(key)

2。session：  是在服务器端进行用户信息的保存。一个字典
注意：
使用session必须要设置配置文件，在配置文件中添加SECRET_KEY='xxxxx'，
添加SECRET_KEY的目的就是用于sessionid的加密。如果不设置会报错。

  设置：
    如果要使用session，需要直接导入：
    from flask import session

    把session当成字典使用，因此：session[key]=value
    就会将key=value保存到session的内存空间
    **** 并会在响应的时候自动在response中自动添加有一个cookie：session=加密后的id ****
  获取
     用户请求页面的时候就会携带上次保存在客户端浏览器的cookie值，其中包含session=加密后的id
     获取session值的话通过session直接获取，因为session是一个字典，就可以采用字典的方式获取即可。
     value = session[key] 或者 value = session.get(key)
     这个时候大家可能会考虑携带的cookie怎么用的？？？？
     其实是如果使用session获取内容,底层会自动获取cookie中的sessionid值，
     进行查找并找到对应的session空间

   删除
    session.clear()  删除session的内存空间和删除cookie
    del session[key]  只会删除session中的这个键值对，不会删除session空间和cookie


secretID：dcc535cbfaefa2a24c1e6610035b6586
secretKey：d28f0ec3bf468baa7a16c16c9474889e
bid ：748c53c3a363412fa963ed3c1b795c65

---- 5.28 -----

1.短信息发送：


2.登录权限的验证
只要走center路由，判断用户是否是登录状态，如果用户登录了，可以正常显示页面，如果用户没有登录
则自动跳转到登录页面进行登录，登录之后才可以进行查看。

钩子函数：
直接应用在app上：
before_first_request
before_request
after_request
teardown_request

应用到蓝图：
before_app_first_request
before_app_request
after_app_request
teardown_app_request

3.文件上传
 A. 本地上传
    注意：
    表单：  enctype="multipart/form-data"
 <form action="提交的路径{{url_for('user.upload_photo')}}" method="post" enctype="multipart/form-data">
        <input type="file" name="photo" class="form-control">
        <input type="submit" value="上传相册" class="btn btn-default">
 </form>
   view视图函数：
   photo = request.files.get('photo')   ----》photo是FileStorage

   属性和方法：FileStorage = 》fs
   fs.filename
   fs.save(path)  ----> path上传的路径os.path.join(UPLOAD_DIR,filename)
   fs.read()  ----> 将上传的内容转成二进制方式

 B. 上传到云端（对象存储）
    本地的资源有限或者是空间是有限的

    https://developer.qiniu.com/kodo/sdk/1242/python  ---》参照python SDK

    util.py:

    def upload_qiniu():
        #需要填写你的 Access Key 和 Secret Key
        access_key = 'Access_Key'
        secret_key = 'Secret_Key'
        #构建鉴权对象
        q = Auth(access_key, secret_key)
        #要上传的空间
        bucket_name = 'Bucket_Name'
        #上传后保存的文件名
        key = 'my-python-logo.png'
        #生成上传 Token，可以指定过期时间等
        token = q.upload_token(bucket_name, key, 3600)
        #要上传文件的本地路径
        localfile = './sync/bbb.jpg'
        ret, info = put_file(token, key, localfile)
        print(info)

        ---->put_data()  适用于从filestorage里面读取数据实现上传
        ---->put_file()  指定文件路径上传

    def delete_qiniu():
        pass

评论：
    文章的详情：必须携带aid，aid表示的是文章的主键id

    通过主键id得到文章对象

    如果还有其他内容的分页，就需要在路由携带page

    例如：http://127.0.0.1:5000/article/detail?page=2&aid=1
