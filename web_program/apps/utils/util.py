import random

from qiniu import Auth, put_file, etag, put_data, BucketManager
import qiniu.config


def upload_qiniu(filestorage):
    # 需要填写你的 Access Key 和 Secret Key
    access_key = '1fXvG9wkbN7AgRUG6usHDcRP5Bb85apcovRAIITP'
    secret_key = 'Aqf1lPAmUG72EdZJ7PxKtWHfWDYNdUycZP1TaAIN'
    # 构建鉴权对象
    q = Auth(access_key, secret_key)
    # 要上传的空间
    bucket_name = 'myblog202006'
    # 上传后保存的文件名
    filename = filestorage.filename
    ran = random.randint(1, 1000)
    suffix = filename.rsplit('.')[-1]
    key = filename.rsplit('.')[0] + '_' + str(ran) + '.' + suffix
    # 生成上传 Token，可以指定过期时间等
    token = q.upload_token(bucket_name, key, 3600)
    # 要上传文件的本地路径
    # localfile = './sync/bbb.jpg'
    #localfile = 'D:/2023毕业设计资料/PycharmProject/ProjectLearn/flask_mysql/flaskday06/static/upload/icon/hdImg_0d42ef6107efdad22aaa97b64de1653516137425630.jpg'
    #ret, info = put_file(token, key, localfile)
    ret, info = put_data(token, key, filestorage.read())
    return ret, info


def delete_qiniu(filename):
    # 需要填写你的 Access Key 和 Secret Key
    access_key = '1fXvG9wkbN7AgRUG6usHDcRP5Bb85apcovRAIITP'
    secret_key = 'Aqf1lPAmUG72EdZJ7PxKtWHfWDYNdUycZP1TaAIN'
    # 构建鉴权对象
    q = Auth(access_key, secret_key)
    # 要上传的空间
    bucket_name = 'myblog202006'
    # 初始化BucketManager
    bucket = BucketManager(q)
    # key就是要删除的文件的名字
    key = filename
    ret, info = bucket.delete(bucket_name, key)
    return info
