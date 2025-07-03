# YOLO智能肿瘤检测系统

基于YOLO深度学习的医学影像智能检测平台，支持用户上传医学图片和自定义模型，自动识别肿瘤类型并生成美观的PDF检测报告，集成AI点评，适合科研、教学和临床辅助。

---

## 主要功能

- 🖼️ **图片上传与检测**：支持用户上传医学影像，选择自定义YOLO模型进行目标检测。
- 🤖 **AI智能点评**：集成大模型API，对检测结果自动生成专业点评。
- 📊 **检测历史管理**：自动保存每次检测记录，支持历史查询与管理。
- 📄 **一键导出PDF报告**：生成美观、专业的中文PDF检测报告，支持AI点评与图片嵌入。
- 👤 **用户注册与登录**：多用户管理，数据隔离。
- 📈 **数据统计与仪表盘**：可视化展示检测次数、用户数、活跃度等。

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/你的用户名/yoloproject.git
cd yoloproject
```

### 2. 安装依赖

建议使用Python 3.8+，推荐虚拟环境。

```bash
pip install -r requirements.txt
```

**主要依赖：**
- Flask
- ultralytics (YOLO)
- reportlab
- pillow
- pymysql
- requests

### 3. 数据库准备

- 使用MySQL，需提前创建数据库`shixi`，并在`yoloweb/app.py`中配置好账号密码。
- 初始化表结构（可根据实际表结构补充SQL）。

### 4. 启动服务

```bash
cd yoloweb
python app.py
```

浏览器访问 [http://localhost:5000](http://localhost:5000)

---

## 目录结构

```
yoloproject/
├── yoloweb/
│   ├── app.py                # Flask主应用
│   ├── static/               # 静态资源（上传、检测图片、字体等）
│   ├── templates/            # 前端页面模板
│   └── models/               # 用户上传的YOLO模型
├── yoloserver/
│   ├── scripts/              # 训练、推理等脚本
│   ├── utils/                # 数据、模型、日志等工具
│   └── ...                   # 其它后端相关
└── requirements.txt          # 依赖列表
```

---

## 常见问题

- **PDF中文乱码？**  
  请确保`yoloweb/static/`目录下有`simhei.ttf`字体文件。

- **导出PDF报错UnicodeEncodeError？**  
  已修复，下载文件名采用英文或RFC 6266标准，兼容中文。

- **YOLO模型格式？**  
  推荐使用ultralytics YOLOv8格式（`.pt`），可在`scripts/`目录下训练或转换。

- **数据库连接失败？**  
  检查MySQL服务、账号密码、数据库名是否正确。

---

## 贡献与交流

欢迎提交Issue、PR或交流建议！  
如需定制开发、模型训练、算法优化等服务，请联系作者。

---

## License

本项目仅供学习与科研使用，禁止用于商业用途。

---

如需英文版README或更详细的开发文档，请联系作者。

如需自动生成requirements.txt或数据库建表SQL，也可告知！
