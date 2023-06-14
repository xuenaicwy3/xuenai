import os

from flask import Blueprint, render_template, request, redirect, url_for, jsonify, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from apps.article.models import Article_type, Article
from apps.user.models import User, Photo
from apps.user.smssend import SmsSendAPIDemo
from apps.utils.util import upload_qiniu, delete_qiniu
from exts import db
from settings import Config

###########################################################
import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from models2.SE_ConvNext_downsample_block import convnext_tiny
#from models2.SE_ConvNext_block import convnext_tiny
app = Flask(__name__)
CORS(app)  # 解决跨域问题
##################################################################

user_bp1 = Blueprint('user', __name__, url_prefix='/user')

required_login_list = ['/user/center', '/user/change', '/article/publish', '/user/upload_photo', '/user/photo_del']

#############################################################################
############################################################################
weights_path = "./SEConvNeXt_downsample_block_epo=40八分类.pth"
class_json_path = "./class_indices.json"
assert os.path.exists(weights_path), "weights path does not exist..."
assert os.path.exists(class_json_path), "class json path does not exist..."

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# create model
#model = densenet121(num_classes=8)
model = convnext_tiny(num_classes=8)
# load model weights
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

# load class info
json_file = open(class_json_path, 'rb')
class_indict = json.load(json_file)


def transform_image(image_bytes):
    img_size = 224
    my_transforms = transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                        transforms.CenterCrop(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        raise ValueError("input file does not RGB image...")
    return my_transforms(image).unsqueeze(0).to(device)


def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = torch.softmax(model.forward(tensor).squeeze(), dim=0)
        prediction = outputs.detach().cpu().numpy()
        template = "诊断结果:{:<15}"
        # template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


@user_bp1.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    image = request.files['photo']
    #image = request.files.get('photo')
    print('======>', image)  # FileStorage
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)


@user_bp1.route("/p", methods=["GET", "POST"])
def root():
    return render_template("up.html")
################################################################################
################################################################################

@user_bp1.before_app_first_request
def first_request():
    print('before_app_first_request')


# ****重点*****
@user_bp1.before_app_request
def before_request1():
    print('before_request1before_request1', request.path)
    if request.path in required_login_list:
        id = session.get('uid')
        if not id:
            return render_template('user/login.html')
        else:
            user = User.query.get(id)
            # g对象，本次请求的对象
            g.user = user


@user_bp1.after_app_request
def after_request_test(response):
    response.set_cookie('a', 'bbbb', max_age=19)
    print('after_request_test')
    return response


@user_bp1.teardown_app_request
def teardown_request_test(response):
    print('teardown_request_test')
    return response


# 自定义过滤器

@user_bp1.app_template_filter('cdecode')
def content_decode(content):
    content = content.decode('utf-8')
    return content[:200]


# 首页
@user_bp1.route('/')
def index():
    # 1。cookie获取方式
    # uid = request.cookies.get('uid', None)
    # 2。session的获取,session底层默认获取
    # 2。session的方式：
    uid = session.get('uid')
    # 获取文章列表   7 6 5  |  4 3 2 | 1
    # 接收页码数
    page = int(request.args.get('page', 1))
    pagination = Article.query.order_by(-Article.pdatetime).paginate(page=page, per_page=3)
    print(pagination.items)  # [<Article 4>, <Article 3>, <Article 2>]
    print(pagination.page)  # 当前的页码数
    print(pagination.prev_num)  # 当前页的前一个页码数
    print(pagination.next_num)  # 当前页的后一页的页码数
    print(pagination.has_next)  # True
    print(pagination.has_prev)  # True
    print(pagination.pages)  # 总共有几页
    print(pagination.total)  # 总的记录条数
    # 获取分类列表
    types = Article_type.query.all()
    # 判断用户是否登录
    if uid:
        user = User.query.get(uid)
        return render_template('user/index.html', user=user, types=types, pagination=pagination)
    else:
        return render_template('user/index.html', types=types, pagination=pagination)


# 用户注册
@user_bp1.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        repassword = request.form.get('repassword')
        phone = request.form.get('phone')
        email = request.form.get('email')
        if password == repassword:
            # 注册用户
            user = User()
            user.username = username
            # 使用自带的函数实现加密：generate_password_hash
            user.password = generate_password_hash(password)
            # print(password)
            user.phone = phone
            user.email = email
            # 添加并提交
            db.session.add(user)
            db.session.commit()
            return redirect(url_for('user.index'))
    return render_template('user/register.html')


# 手机号码验证
@user_bp1.route('/checkphone', methods=['GET', 'POST'])
def check_phone():
    phone = request.args.get('phone')
    user = User.query.filter(User.phone == phone).all()
    print(user)
    # code: 400 不能用    200 可以用
    if len(user) > 0:
        return jsonify(code=400, msg='此号码已被注册')
    else:
        return jsonify(code=200, msg='此号码可用')


# 用户登录
@user_bp1.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        f = request.args.get('f')
        if f == '1':  # 用户名或者密码
            username = request.form.get('username')
            password = request.form.get('password')
            users = User.query.filter(User.username == username).all()
            for user in users:
                # 如果flag=True表示匹配，否则密码不匹配
                flag = check_password_hash(user.password, password)
                if flag:
                    # 1。cookie实现机制
                    # response = redirect(url_for('user.index'))
                    # response.set_cookie('uid', str(user.id), max_age=1800)
                    # return response
                    # 2。session机制,session当成字典使用
                    session['uid'] = user.id
                    return redirect(url_for('user.index'))
            else:
                return render_template('user/login.html', msg='用户名或者密码有误')
        elif f == '2':  # 手机号码与验证码
            print('----->22222')
            phone = request.form.get('phone')
            code = request.form.get('code')
            # 先验证验证码
            valid_code = session.get(phone)
            print('valid_code:' + str(valid_code))
            if code == valid_code:
                # 查询数据库
                user = User.query.filter(User.phone == phone).first()
                print(user)
                if user:
                    # 登录成功
                    session['uid'] = user.id
                    return redirect(url_for('user.index'))
                else:
                    return render_template('user/login.html', msg='此号码未注册')
            else:
                return render_template('user/login.html', msg='验证码有误！')

    return render_template('user/login.html')


# 发送短信息
@user_bp1.route('/sendMsg')
def send_message():
    phone = request.args.get('phone')
    # 补充验证手机号码是否注册，去数据库查询

    SECRET_ID = "dcc535cbfaefa2a24c1e6610035b6586"  # 产品密钥ID，产品标识
    SECRET_KEY = "d28f0ec3bf468baa7a16c16c9474889e"  # 产品私有密钥，服务端生成签名信息使用，请严格保管，避免泄露
    BUSINESS_ID = "748c53c3a363412fa963ed3c1b795c65"  # 业务ID，易盾根据产品业务特点分配
    api = SmsSendAPIDemo(SECRET_ID, SECRET_KEY, BUSINESS_ID)
    params = {
        "mobile": phone,
        "templateId": "11732",
        "paramType": "json",
        "params": "json格式字符串"
    }
    ret = api.send(params)
    print(ret)
    session[phone] = '666666'
    return jsonify(code=200, msg='短信发送成功！')

    # if ret is not None:
    #     if ret["code"] == 200:
    #         taskId = ret["result"]["taskId"]
    #         print("taskId = %s" % taskId)
    #         session[phone] = '189075'
    #         return jsonify(code=200, msg='短信发送成功！')
    #     else:
    #         print("ERROR: ret.code=%s,msg=%s" % (ret['code'], ret['msg']))
    #         return jsonify(code=400, msg='短信发送失败！')


# 用户退出
@user_bp1.route('/logout')
def logout():
    # 1。 cookie的方式
    # response = redirect(url_for('user.index'))
    # 通过response对象的delete_cookie(key),key就是要删除的cookie的key
    # response.delete_cookie('uid')
    # 2。session的方式
    # del session['uid']
    session.clear()
    return redirect(url_for('user.index'))


# 用户中心
@user_bp1.route('/center')
def user_center():
    types = Article_type.query.all()
    photos = Photo.query.filter(Photo.user_id == g.user.id).all()
    return render_template('user/center.html', user=g.user, types=types, photos=photos)


# 图片的扩展名
ALLOWED_EXTENSIONS = ['jpg', 'png', 'JPG', 'PNG', 'gif', 'bmp', 'jpeg']


# 用户信息修改
@user_bp1.route('/change', methods=['GET', 'POST'])
def user_change():
    if request.method == 'POST':
        username = request.form.get('username')
        phone = request.form.get('phone')
        email = request.form.get('email')
        # 只要有文件（图片），获取方式必须使用request.files.get(name)
        icon = request.files.get('icon')
        # print('======>', icon)  # FileStorage
        # 属性： filename 用户获取文件的名字
        # 方法:  save(保存路径)
        icon_name = icon.filename  # 1440w.jpg
        suffix = icon_name.rsplit('.')[-1]
        if suffix in ALLOWED_EXTENSIONS:
            icon_name = secure_filename(icon_name)  # 保证文件名是符合python的命名规则
            file_path = os.path.join(Config.UPLOAD_ICON_DIR, icon_name)
            icon.save(file_path)
            # 保存成功
            user = g.user
            user.username = username
            user.phone = phone
            user.email = email
            path = 'upload/icon/'
            user.icon = os.path.join(path, icon_name)
            db.session.commit()
            return redirect(url_for('user.user_center'))
        else:
            return render_template('user/center.html', user=g.user, msg='必须是扩展名是：jpg,png,gif,bmp,jpeg格式')

        # 查询一下手机号码
        # users = User.query.all()
        # for user in users:
        #     if user.phone == phone:
        #         # 说明数据库中已经有人注册此号码
        #         return render_template('user/center.html', user=g.user,msg='此号码已被注册')
        #

    return render_template('user/center.html', user=g.user)


# 发表文章
@user_bp1.route('/article', methods=['GET', 'POST'])
def publish_article():
    if request.method == 'POST':
        title = request.form.get('title')
        type = request.form.get('type')
        content = request.form.get('content')
        print(title, type, content)

        return render_template('article/test.html', content=content)
    return '发表失败！'


# 上传照片
@user_bp1.route('/upload_photo', methods=['GET', 'POST'])
@torch.no_grad()
def upload_photo():
    # 获取上传的内容
    photo = request.files.get('photo')  # FileStorage
    print('======>', photo)  # FileStorage
    # 属性： filename 用户获取文件的名字
    # 方法:  save(保存路径)
    # photo.filename,photo.save(path)
    photo_name = photo.filename
    suffix = photo_name.rsplit('.')[-1]
    if suffix in ALLOWED_EXTENSIONS:
        photo_name = secure_filename(photo_name)
        file_path = os.path.join(Config.UPLOAD_PHOTO_DIR, photo_name)
        photo.save(file_path)
        # 保存成功
        path = 'upload/photo/'
        photo = Photo()
        #photo.photo_name = os.path.join(photo_name)
        photo.photo_name = os.path.join(path, photo_name)
        print('================>', photo.photo_name)
        photo.user_id = g.user.id
        #db.session.add(photo)
        db.session.commit()
        return redirect(url_for('user.user_center'))
    else:
         return '上传失败！'
    # image = request.files['photo']
    # #image = request.files.get('photo')
    # print('======>', image)  # FileStorage
    # img_bytes = image.read()
    # info = get_prediction(image_bytes=img_bytes)
    # return jsonify(info)
    # 工具模块中封装方法
    # ret, info = upload_qiniu(photo)
    # if info.status_code == 200:
    #     photo = Photo()
    #     photo.photo_name = ret['key']
    #     photo.user_id = g.user.id
    #     db.session.add(photo)
    #     db.session.commit()
    #     return '上传成功！'
    # else:
    #     return '上传失败！'


# 删除相册图片
@user_bp1.route('/photo_del')
def photo_del():
    pid = request.args.get('pid')
    photo = Photo.query.get(pid)
    filename = photo.photo_name
    print('filename=', filename)
    # 封装好的一个删除七牛存储文件的函数
    info = delete_qiniu(filename)
    # 判断状态码
    #if info.status_code == 200:
        # 删除数据库的内容
    db.session.delete(photo)
    db.session.commit()
    return redirect(url_for('user.user_center'))
    # else:
    #
    #     return render_template('500.html', err_msg='删除相册图片失败！')
@user_bp1.route('/photo_del1')
def photo_del1():
    pid = request.args.get('pid')
    photo = Photo.query.get(pid)
    filename = photo.photo_name
    print('filename=', filename)
    # 封装好的一个删除七牛存储文件的函数
    info = delete_qiniu(filename)
    # 判断状态码
    #if info.status_code == 200:
        # 删除数据库的内容
    db.session.delete(photo)
    db.session.commit()
    return redirect(url_for('user.myphoto'))
    # else:
    #
    #     return render_template('500.html', err_msg='删除相册图片失败！')


# 诊断记录
@user_bp1.route('/myphoto')
def myphoto():
    # photos = Photo.query.all()
    page = int(request.args.get('page', 1))
    # # 分页操作
    photos = Photo.query.paginate(page=page, per_page=3)
    user_id = session.get('uid',None)
    user = None
    if user_id:
        user = User.query.get(user_id)
    return render_template('user/myphoto.html', photos=photos, user=user)


# 图像信息修改
@user_bp1.route('/update',  methods=['GET', 'POST'])
def photo_update():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        photo_name = request.form.get('photo_name')
        photo_datetime = request.form.get('photo_datetime')
        photo_predict = request.form.get('photo_predict')
        photo_advise = request.form.get('photo_advise')
        id = request.form.get('id')
        # 保存成功
        #photo = Photo()
        photo = Photo.query.get(id)
        photo.user_id = user_id
        photo.photo_name = photo_name
        photo.photo_datetime = photo_datetime
        photo.photo_predict = photo_predict
        photo.photo_advise = photo_advise
        db.session.commit()
        return redirect(url_for('user.user_center'))

    else:
        id = request.args.get('id')
        photo = Photo.query.get(id)
        return render_template('user/center.html', photo=photo)


@user_bp1.route('/error')
def test_error():
    # print(request.headers)
    # print(request.headers.get('Accept-Encoding'))
    referer = request.headers.get('Referer', None)
    return render_template('500.html', err_msg='有误', referer=referer)
