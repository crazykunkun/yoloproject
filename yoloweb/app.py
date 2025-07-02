import os
from flask import Flask, request, render_template, jsonify
import sys
from ultralytics import YOLO
from PIL import Image
import numpy as np
from datetime import datetime
import requests
import json


API_KEY = "sk-nnmesdasavgfaksjicfuuwcoqugdrgcgsgiktvdhowelsymq"
app = Flask(__name__)

# 关闭浏览器缓存
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# 设置上传与检测目录
UPLOAD_FOLDER = 'yoloweb/static/uploads'
DETECT_FOLDER = 'yoloweb/static/detections'
MODELS_FOLDER = 'yoloweb/static/models'  # 新增模型上传目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECT_FOLDER'] = DETECT_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER

HISTORY_FILE = 'yoloweb/static/history/history.json'
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

def save_history_record(record):
    # 读取历史
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = []
    # 追加新记录
    history.append(record)
    # 保存
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def get_ai_comment(detections, token):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    # 拼接检测结果
    det_text = "，".join(detections)
    prompt = f"你是一个肿瘤专家，这是yolo识别的结果：{det_text}，用一句话点评，50字左右。不要任何其他的废话"
    payload = {
        "model": "Qwen/QwQ-32B",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {token}",  
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        data = response.json()
        print("AI原始返回：", data)
        # 解析AI返回内容
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI点评获取失败: {e}"

@app.route('/', methods=['GET', 'POST'])
def upload_detect():
    if request.method == 'POST':
        # 获取客户端上传的图片
        image_file = request.files["image"]
        # 获取客户端上传的模型
        model_file = request.files["model"]

        if image_file and model_file:
            # 处理上传的图片
            img_filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + image_file.filename
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], img_filename)
            detect_path = os.path.join(app.config["DETECT_FOLDER"], img_filename)
            image_file.save(upload_path)

            # 处理上传的模型
            model_filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + model_file.filename
            model_path = os.path.join(app.config["MODELS_FOLDER"], model_filename)
            model_file.save(model_path)

            try:
                # 使用上传的模型进行目标检测
                model = YOLO(model_path)

                # 进行目标检测
                results = model(upload_path)

                # 绘制检测结果图像并保存
                result_img_array = results[0].plot()
                result_pil = Image.fromarray(result_img_array)
                result_pil.save(detect_path)
                print("图片保存路径：", detect_path, "是否存在：", os.path.exists(detect_path))

                # 提取检测框信息（标签 + 置信度）
                detections = []
                tumor_types = set()
                boxes = results[0].boxes
                if boxes is not None and boxes.cls.numel() > 0:
                    for cls_id, conf in zip(boxes.cls, boxes.conf):
                        class_name = model.names[int(cls_id)]
                        confidence = round(float(conf) * 100, 2)
                        detections.append(f"{class_name}: {confidence}%")
                        tumor_types.add(class_name)
                else:
                    detections.append("No objects detected.")

                # === 新增AI点评 ===
                ai_comment = get_ai_comment(detections, API_KEY) 

                record = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_file": img_filename,
                    "detect_file": img_filename,  # 检测结果图片和原图同名
                    "model_file": model_filename,
                    "detections": detections,
                    "tumor_types": list(tumor_types) if tumor_types else ["无"],
                    "ai_comment": ai_comment  # 新增点评
                }
                save_history_record(record)

                return render_template(
                    'index.html',
                    prediction="Detection Complete",
                    detections=detections,
                    image_path=f"detections/{img_filename}",
                    ai_comment=ai_comment  # 新增
                )
            except Exception as e:
                return render_template(
                    'index.html',
                    prediction=f"Error: {str(e)}",
                    detections=[],
                    image_path=None
                )

    return render_template('index.html', prediction=None)

@app.route('/history')
def history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = []
    return render_template('history.html', history=history)

@app.route('/refresh_ai_comment', methods=['POST'])
def refresh_ai_comment():
    data = request.get_json()
    detections = data.get("detections", [])
    # 重新获取AI点评
    ai_comment = get_ai_comment(detections, API_KEY)
    return jsonify({"ai_comment": ai_comment})

if __name__ == '__main__':
    app.run(debug=True)