
# -*- coding: UTF-8 -*-
import os
import shutil
# request フォームから送信した情報を扱うためのモジュール
# redirect  ページの移動
# url_for アドレス遷移
from flask import Flask, request, redirect, url_for, render_template, flash
# ファイル名をチェックする関数
from werkzeug.utils import secure_filename
# 画像のダウンロード
from flask import send_from_directory
import numpy as np
import cv2
import sys
from statistics import mean
from datetime import datetime
import string

app = Flask(__name__)


app.secret_key = os.getenv('SECRET_KEY', 'for dev')
SAVE_DIR = "./images"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

IMG_DIR = './images_stock/'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = SAVE_DIR

estimated_d =[]

names = ["beige","black","blue","red","yellow","green","gray","check","pink","multi_tone","Polka_dot","brown","unicro","actress"]
exists_img=[]
img_url=""

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory(SAVE_DIR, path)


# ファイルを受け取る方法の指定
@app.route('/', methods=['GET','POST'])
def index():
    
    if exists_img ==[]:
        estimated_d=[]
        return render_template("index.html",names=names)
    else:
        return render_template("index.html", names=names,data= zip(exists_img,estimated_d))


    



@app.route('/upload', methods=['GET','POST'])
def upload():
    shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)


    # if request.method == 'POST':
    # name = request.form.get('names')
    name = request.form['names']

    
    name = str(name)
    if name == "beige":
        SUB_DIR = 'beige/'
    elif name == "black":
        SUB_DIR = 'black/'
    elif name == 'blue':
        SUB_DIR = 'blue/'
    elif name == 'brown':
        SUB_DIR = 'brown/'
    elif name == "check":
        SUB_DIR = 'check/'
    elif name == "gray":
        SUB_DIR = 'gray/'
    elif name == "green":
        SUB_DIR = 'green/'
    elif name == "multi_tone":
        SUB_DIR = 'multi_tone/'
    elif name == "pink":
        SUB_DIR = 'pink/'
    elif name == "red":
        SUB_DIR = 'red/'
    elif name == "Polka_dot":
        SUB_DIR = 'Polka_dot/'
    elif name == "yellow":
        SUB_DIR = 'yellow/'
    elif name == "actress":
        SUB_DIR = 'actress/'
    else:
        SUB_DIR = 'unicro/'



    # # ファイルがなかった場合の処理
    # if 'file' not in request.files:
    #     flash('ファイルがありません','failed')
    #     return redirect(request.url)
    # file = request.files['image']
    #             # ファイルのチェック
    # if file and allowed_file(file.filename):
        # 危険な文字を削除（サニタイズ処理）
        # filename = secure_filename(file.filename)
    # 画像として読み込み
    stream = request.files['image'].stream
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    img_size = (200, 200)
    channels = (0, 1, 2)
    mask = None
    hist_size = 256
    ranges = (0, 256)
    ret = {}


    
    if SUB_DIR != 'actress/':
        target_img = img
        target_img = cv2.resize(target_img, img_size)
        # if target_img and allowed_file(target_img.filename):
        #     filename = secure_filename(target_img.filename)
        #     target_img.save(os.path.join('/uploads', filename))
        #     img_url = '/uploads/' + filename
        
        
        comparing_files = os.listdir(IMG_DIR + SUB_DIR)

 
 
        
        if len(comparing_files) == 0:
            sys.exit(1)




        for comparing_file in comparing_files:
            if comparing_file == '.DS_Store':
                continue
        
            tmp = []
            if not comparing_file.endswith(('.png', '.jpg', '.jpeg')):
                continue

            for channel in channels:
                target_hist = cv2.calcHist([target_img], [channel], mask, [hist_size], ranges)
                comparing_img = cv2.imread(IMG_DIR + SUB_DIR + comparing_file)
                
                comparing_img = cv2.resize(comparing_img, img_size)
                
                # calc hist of comparing image
                comparing_hist = cv2.calcHist([comparing_img], [channel], mask, [hist_size], ranges)

                # compare hist
                tmp.append(cv2.compareHist(target_hist, comparing_hist, 0))

            # mean hist
            ret[comparing_file] = mean(tmp)

            # sort
        
        



    #####################################3

    if SUB_DIR == 'actress/':

        # if img and allowed_file(img.filename):
        #     filename = secure_filename(img.filename)
        #     img.save(os.path.join('/uploads', filename))
        #     img_url = '/uploads/' + filename
    
        target_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        target_img = cv2.resize(target_img, img_size)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # detector = cv2.ORB_create()
        detector = cv2.AKAZE_create()
        (_, target_des) = detector.detectAndCompute(target_img, None)

        comparing_files = os.listdir(IMG_DIR + SUB_DIR)

        for comparing_file in comparing_files:
            if comparing_file == '.DS_Store':
                continue

            comparing_img_path = IMG_DIR + SUB_DIR + comparing_file
            try:
                comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
                comparing_img = cv2.resize(comparing_img, img_size)
                (_, comparing_des) = detector.detectAndCompute(comparing_img, None)
                matches = bf.match(target_des, comparing_des)
                dist = [m.distance for m in matches]
                score = sum(dist) / len(dist)
                if score <= 1:
                    score = 1
                score = 100.0 / score
            except cv2.error:
                score = 100000

            ret[comparing_file] = score

    ############################################################
    
    
    
    dic_sorted = sorted(ret.items(), reverse=True,key=lambda x:x[1])[:3]
    estimated_d=[]
    for file in dic_sorted:
        img_path = IMG_DIR + SUB_DIR + file[0]
        img = cv2.imread(img_path)
        # cv2.imshow('image',img)
                # 保存
        dt_now = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_%f")
        save_path = os.path.join(SAVE_DIR, dt_now + ".jpeg")
        cv2.imwrite(save_path, img)
        estimated_d.append(file[1])
    f_imgs = os.listdir(SAVE_DIR)
    if '.DS_Store' in f_imgs:
        f_imgs.remove('.DS_Store')
    exists_img = sorted(f_imgs)[-3:]
        

    

    # ファイルの保存
    # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # アップロード後のページに転送
    return render_template('index.html',names=names,data= zip(exists_img,estimated_d))


if __name__ == '__main__':
    app.debug = True
    app.run()

    



    





