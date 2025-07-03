import os
from flask import Flask, request, render_template, jsonify, session, redirect
import sys
from ultralytics import YOLO
from PIL import Image
import numpy as np
from datetime import datetime
import requests
import json
import pymysql
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm


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

# MySQL配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'shixi',
    'charset': 'utf8mb4'
}

def get_db_conn():
    return pymysql.connect(**DB_CONFIG)

def save_history_record(record, user_id, ip=None):
    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO history (user_id, time, image_file, detect_file, model_file, detections, tumor_types, ai_comment, ip) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    user_id,
                    record["time"],
                    record["image_file"],
                    record["detect_file"],
                    record["model_file"],
                    json.dumps(record["detections"], ensure_ascii=False),
                    json.dumps(record["tumor_types"], ensure_ascii=False),
                    record["ai_comment"],
                    ip
                )
            )
            conn.commit()
    finally:
        conn.close()

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

app.secret_key = '123456'

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            return render_template('register.html', error='用户名和密码不能为空')
        conn = get_db_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute('SELECT id FROM users WHERE username=%s', (username,))
                if cursor.fetchone():
                    return render_template('register.html', error='用户名已存在')
                password_hash = generate_password_hash(password)
                cursor.execute('INSERT INTO users (username, password_hash) VALUES (%s, %s)', (username, password_hash))
                conn.commit()
            return redirect('/login')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        conn = get_db_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute('SELECT id, password_hash FROM users WHERE username=%s', (username,))
                user = cursor.fetchone()
                if user and check_password_hash(user[1], password):
                    session['user_id'] = user[0]
                    session['username'] = username
                    return redirect('/')
                else:
                    return render_template('login.html', error='用户名或密码错误')
        finally:
            conn.close()
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/', methods=['GET', 'POST'])
@login_required
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
                save_history_record(record, session['user_id'], request.remote_addr)

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
@login_required
def history():
    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT time, image_file, detect_file, model_file, detections, tumor_types, ai_comment "
                "FROM history WHERE user_id=%s ORDER BY time DESC", (session['user_id'],)
            )
            rows = cursor.fetchall()
            history = []
            for row in rows:
                history.append({
                    "time": row[0].strftime("%Y-%m-%d %H:%M:%S"),
                    "image_file": row[1],
                    "detect_file": row[2],
                    "model_file": row[3],
                    "detections": json.loads(row[4]),
                    "tumor_types": json.loads(row[5]),
                    "ai_comment": row[6]
                })
    finally:
        conn.close()
    return render_template('history.html', history=history)

@app.route('/refresh_ai_comment', methods=['POST'])
def refresh_ai_comment():
    data = request.get_json()
    detections = data.get("detections", [])
    # 重新获取AI点评
    ai_comment = get_ai_comment(detections, API_KEY)
    return jsonify({"ai_comment": ai_comment})

@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            # 总调用次数
            cursor.execute("SELECT COUNT(*) FROM history")
            total_calls = cursor.fetchone()[0]
            # 总用户数
            cursor.execute("SELECT COUNT(*) FROM users")
            total_users = cursor.fetchone()[0]
            # 处理总量（图片数）
            cursor.execute("SELECT COUNT(*) FROM history")
            total_processed = cursor.fetchone()[0]
            # 近一周每天调用次数
            cursor.execute("""
                SELECT DATE(time), COUNT(*) FROM history
                WHERE time >= DATE_SUB(CURDATE(), INTERVAL 6 DAY)
                GROUP BY DATE(time)
                ORDER BY DATE(time)
            """)
            week_data = cursor.fetchall()
            week_labels = [str(row[0]) for row in week_data]
            week_counts = [row[1] for row in week_data]
            # 最近10个IP
            cursor.execute("SELECT ip, time FROM history WHERE ip IS NOT NULL ORDER BY time DESC LIMIT 10")
            ip_records = cursor.fetchall()
    finally:
        conn.close()
    return render_template(
        'dashboard.html',
        total_calls=total_calls,
        total_users=total_users,
        total_processed=total_processed,
        week_labels=week_labels,
        week_counts=week_counts,
        ip_records=ip_records
    )

# 注册中文字体（假设simhei.ttf在static目录下）
pdfmetrics.registerFont(TTFont('simhei', os.path.join('yoloweb', 'static', 'simhei.ttf')))

@app.route('/export_pdf', methods=['POST'])
@login_required
def export_pdf():
    data = request.get_json()
    detections = data.get('detections', [])
    ai_comment = data.get('ai_comment', '')
    image_path = data.get('image_path', '')
    username = session.get('username', '')
    detect_time = data.get('detect_time', '')

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    # 定义现代化的样式
    styles.add(ParagraphStyle(
        name='ReportTitle',
        fontName='simhei',
        fontSize=32,
        alignment=1,
        spaceAfter=15,
        leading=40,
        textColor=colors.HexColor('#1a202c'),
        borderPadding=10
    ))
    
    styles.add(ParagraphStyle(
        name='ReportSubTitle',
        fontName='simhei',
        fontSize=16,
        alignment=1,
        textColor=colors.HexColor('#4a5568'),
        spaceAfter=25,
        leading=20
    ))
    
    styles.add(ParagraphStyle(
        name='ReportInfo',
        fontName='simhei',
        fontSize=12,
        textColor=colors.HexColor('#2d3748'),
        leading=20,
        spaceAfter=5
    ))
    
    styles.add(ParagraphStyle(
        name='ReportSectionTitle',
        fontName='simhei',
        fontSize=18,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=20,
        spaceAfter=12,
        leading=22,
        borderPadding=5
    ))
    
    styles.add(ParagraphStyle(
        name='ReportAIBox',
        fontName='simhei',
        fontSize=13,
        textColor=colors.HexColor('#1a202c'),
        backColor=colors.HexColor('#e6fffa'),
        borderPadding=15,
        leading=20,
        spaceBefore=15,
        spaceAfter=15,
        borderRadius=8,
        borderColor=colors.HexColor('#38b2ac'),
        borderWidth=1
    ))
    
    styles.add(ParagraphStyle(
        name='ReportFooter',
        fontName='simhei',
        fontSize=10,
        alignment=1,
        textColor=colors.HexColor('#718096'),
        spaceBefore=30,
        leading=15
    ))
    
    styles.add(ParagraphStyle(
        name='ReportHighlight',
        fontName='simhei',
        fontSize=12,
        textColor=colors.HexColor('#2b6cb0'),
        leading=18
    ))

    elements = []
    
    # 页眉装饰
    elements.append(HRFlowable(width="100%", thickness=4, color=colors.HexColor('#4299e1'), spaceBefore=0, spaceAfter=20))
    
    # 标题区域
    elements.append(Paragraph('🔬 智能检测报告', styles['ReportTitle']))
    elements.append(Paragraph('YOLO深度学习识别系统', styles['ReportSubTitle']))
    elements.append(Spacer(1, 15))
    
    # 装饰性分割线
    elements.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#bee3f8'), spaceBefore=0, spaceAfter=25))
    
    # 报告概览卡片
    overview_data = [
        ['📊 报告概览', ''],
        ['👤 检测用户', username],
        ['🕒 检测时间', detect_time],
        ['🎯 检测结果', f'{len(detections)} 个对象' if detections and 'No objects detected' not in str(detections) else '未检测到对象'],
        ['🤖 AI状态', '已生成智能点评' if ai_comment else '无AI点评']
    ]
    
    overview_table = Table(overview_data, colWidths=[45*mm, 115*mm], hAlign='LEFT')
    overview_table.setStyle(TableStyle([
        # 标题行样式
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4299e1')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'simhei'),
        ('FONTSIZE', (0,0), (-1,0), 14),
        ('FONTNAME', (0,0), (0,0), 'simhei'),
        ('SPAN', (0,0), (-1,0)),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        
        # 数据行样式
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f7fafc')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#2d3748')),
        ('FONTNAME', (0,1), (-1,-1), 'simhei'),
        ('FONTSIZE', (0,1), (-1,-1), 11),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f7fafc'), colors.HexColor('#edf2f7')]),
        
        # 边框和内边距
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(overview_table)
    elements.append(Spacer(1, 25))
    
    # 检测图片区域
    if image_path:
        img_abs = os.path.join('yoloweb', 'static', image_path.replace('detections/', 'detections\\'))
        if os.path.exists(img_abs):
            elements.append(Paragraph('📸 检测图片', styles['ReportSectionTitle']))
            elements.append(Spacer(1, 8))
            
            # 图片容器
            img_container_data = [['检测结果图像']]
            img_container = Table(img_container_data, colWidths=[160*mm], hAlign='CENTER')
            img_container.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0f8ff')),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#4299e1')),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,-1), 'simhei'),
                ('FONTSIZE', (0,0), (-1,-1), 12),
                ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#2b6cb0')),
                ('TOPPADDING', (0,0), (-1,-1), 10),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ]))
            elements.append(img_container)
            elements.append(Spacer(1, 8))
            
            # 图片
            img = RLImage(img_abs, width=120*mm, height=120*mm)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 20))
    
    # 识别结果区域
    elements.append(Paragraph('🎯 识别结果', styles['ReportSectionTitle']))
    elements.append(Spacer(1, 8))
    
    if detections and 'No objects detected' not in str(detections):
        # 美化的结果表格
        table_data = [['🏷️ 检测类别', '📊 置信度', '📈 准确性评估']]
        
        for det in detections:
            if ':' in det:
                cls, conf = det.split(':', 1)
                cls = cls.strip()
                conf = conf.strip()
                
                # 添加准确性评估
                conf_num = float(conf.replace('%', ''))
                if conf_num >= 80:
                    accuracy = '🟢 高'
                elif conf_num >= 60:
                    accuracy = '🟡 中'
                else:
                    accuracy = '🔴 低'
                
                table_data.append([cls, conf, accuracy])
            else:
                table_data.append([det, '', ''])
        
        result_table = Table(table_data, colWidths=[60*mm, 40*mm, 50*mm], hAlign='CENTER')
        result_table.setStyle(TableStyle([
            # 标题行
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4299e1')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'simhei'),
            ('FONTSIZE', (0,0), (-1,0), 13),
            ('ALIGN', (0,0), (-1,0), 'CENTER'),
            
            # 数据行
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#2d3748')),
            ('FONTNAME', (0,1), (-1,-1), 'simhei'),
            ('FONTSIZE', (0,1), (-1,-1), 11),
            ('ALIGN', (0,1), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            
            # 交替行背景
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
            
            # 边框
            ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor('#4299e1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
            
            # 内边距
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        elements.append(result_table)
    else:
        # 无检测结果的美化显示
        no_result_data = [['⚠️ 检测状态', '未检测到目标对象']]
        no_result_table = Table(no_result_data, colWidths=[50*mm, 100*mm], hAlign='CENTER')
        no_result_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fed7d7')),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#9b2c2c')),
            ('FONTNAME', (0,0), (-1,-1), 'simhei'),
            ('FONTSIZE', (0,0), (-1,-1), 12),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#fc8181')),
            ('TOPPADDING', (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ]))
        elements.append(no_result_table)
    
    elements.append(Spacer(1, 20))
    
    # AI点评区域
    if ai_comment:
        elements.append(Paragraph('🧠 AI智能点评', styles['ReportSectionTitle']))
        elements.append(Spacer(1, 8))
        
        # AI点评卡片
        ai_content = f"""
        <b>💡 专业分析：</b><br/>
        {ai_comment}
        <br/><br/>
        <i>⚡ 本分析由AI模型自动生成，仅供参考</i>
        """
        
        elements.append(Paragraph(ai_content, styles['ReportAIBox']))
    
    # 页脚区域
    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#e2e8f0'), spaceBefore=0, spaceAfter=15))
    
    footer_text = f"""
    <b>🏥 YOLO智能识别系统</b> | 
    📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
    🔒 数据安全保护
    """
    elements.append(Paragraph(footer_text, styles['ReportFooter']))
    
    # 技术信息
    tech_info = "🔬 基于深度学习YOLO算法 | 🎯 高精度目标检测 | 📊 智能数据分析"
    elements.append(Paragraph(tech_info, styles['ReportFooter']))
    
    doc.build(elements)
    buffer.seek(0)
    return (buffer.getvalue(), 200, {
        'Content-Type': 'application/pdf',
        'Content-Disposition': 'attachment; filename=\"report.pdf\"'
    })


if __name__ == '__main__':
    app.run(debug=True)