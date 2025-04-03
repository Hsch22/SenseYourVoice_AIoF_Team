#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright (c) 2023. All Rights Reserved.

import os
import sys
import argparse
import torch
import tempfile
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# 导入主应用
from app_new import SenseYourVoiceApp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 全局应用实例
sense_app = None

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SenseYourVoice - 语音理解与处理</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            color: #3498db;
            margin-top: 20px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="file"], textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        textarea {
            height: 100px;
            resize: vertical;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #2980b9;
        }
        .result {
            margin-top: 30px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 4px;
            border-left: 4px solid #3498db;
        }
        .result h3 {
            margin-top: 0;
            color: #3498db;
        }
        .error {
            color: #e74c3c;
            font-weight: bold;
        }
        .loading {
            text-align: center;
            display: none;
        }
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border-left-color: #3498db;
            animation: spin 1s linear infinite;
            display: inline-block;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>SenseYourVoice - 语音理解与处理</h1>
        
        <div class="form-group">
            <label for="audio">上传音频文件 (MP3, WAV, 最大16MB):</label>
            <input type="file" id="audio" name="audio" accept="audio/*">
        </div>
        
        <div class="form-group">
            <label for="instruction">处理指令 (可选):</label>
            <textarea id="instruction" name="instruction" placeholder="输入特定的处理指令，例如：'总结这段语音的主要内容'或'分析这段语音中的情感'等"></textarea>
        </div>
        
        <button id="submit-btn" onclick="processAudio()">开始处理</button>
        
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p>正在处理中，请稍候...</p>
        </div>
        
        <div id="result" class="result" style="display: none;">
            <h3>处理结果</h3>
            <div id="transcription-section">
                <h4>语音转文字结果:</h4>
                <p id="transcription"></p>
            </div>
            
            <div id="understanding-section">
                <h4>理解分析结果:</h4>
                <p id="understanding"></p>
            </div>
            
            <div id="specialized-section" style="display: none;">
                <h4>专业任务处理结果:</h4>
                <p id="specialized-result"></p>
            </div>
        </div>
        
        <div id="error" class="error" style="display: none;"></div>
    </div>
    
    <script>
        function processAudio() {
            const audioFile = document.getElementById('audio').files[0];
            const instruction = document.getElementById('instruction').value;
            
            if (!audioFile) {
                document.getElementById('error').textContent = '请选择一个音频文件';
                document.getElementById('error').style.display = 'block';
                return;
            }
            
            const formData = new FormData();
            formData.append('audio', audioFile);
            formData.append('instruction', instruction);
            
            // 显示加载中
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('submit-btn').disabled = true;
            
            fetch('/process', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // 隐藏加载中
                document.getElementById('loading').style.display = 'none';
                document.getElementById('submit-btn').disabled = false;
                
                if (data.success) {
                    // 显示结果
                    document.getElementById('transcription').textContent = data.transcription;
                    document.getElementById('understanding').textContent = data.understanding;
                    
                    if (data.specialized_result) {
                        document.getElementById('specialized-result').textContent = data.specialized_result;
                        document.getElementById('specialized-section').style.display = 'block';
                    } else {
                        document.getElementById('specialized-section').style.display = 'none';
                    }
                    
                    document.getElementById('result').style.display = 'block';
                } else {
                    // 显示错误
                    document.getElementById('error').textContent = '处理失败: ' + data.error;
                    document.getElementById('error').style.display = 'block';
                }
            })
            .catch(error => {
                // 隐藏加载中，显示错误
                document.getElementById('loading').style.display = 'none';
                document.getElementById('submit-btn').disabled = false;
                document.getElementById('error').textContent = '请求错误: ' + error.message;
                document.getElementById('error').style.display = 'block';
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/process', methods=['POST'])
def process_audio():
    global sense_app
    
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': '没有上传音频文件'})
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'success': False, 'error': '未选择文件'})
    
    instruction = request.form.get('instruction', '')
    
    # 保存上传的文件
    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(file_path)
    
    try:
        # 处理音频文件
        result = sense_app.process(file_path, instruction)
        
        # 删除临时文件
        os.remove(file_path)
        
        return jsonify(result)
    except Exception as e:
        # 删除临时文件
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({'success': False, 'error': f'处理过程发生错误: {str(e)}'})

def parse_args():
    parser = argparse.ArgumentParser(description="SenseYourVoice Web界面")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=5000, help="服务器端口")
    parser.add_argument("--model_dir", type=str, default="iic/SenseVoiceSmall", help="语音模型目录")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--understanding_api_key", type=str, help="理解模块API密钥")
    parser.add_argument("--understanding_api_url", type=str, help="理解模块API地址")
    parser.add_argument("--specialized_api_key", type=str, help="专业任务模块API密钥")
    parser.add_argument("--specialized_api_url", type=str, help="专业任务模块API地址")
    return parser.parse_args()

def main():
    global sense_app
    
    args = parse_args()
    
    config = {
        "model_dir": args.model_dir,
        "device": args.device,
        "understanding_api_key": args.understanding_api_key,
        "understanding_api_url": args.understanding_api_url,
        "specialized_api_key": args.specialized_api_key,
        "specialized_api_url": args.specialized_api_url
    }
    
    # 初始化应用
    sense_app = SenseYourVoiceApp(config)
    
    # 启动Flask服务器
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()