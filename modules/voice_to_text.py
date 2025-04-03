#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright (c) 2023. All Rights Reserved.

import os
import sys
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

class VoiceToTextModule:
    def __init__(self, model_dir="iic/SenseVoiceSmall", device="cuda:0" if torch.cuda.is_available() else "cpu"):
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self.initialize_model()
        
    def initialize_model(self):
        try:
            # 修改模型初始化方式，与 demo1.py 或 webui.py 保持一致
            self.model = AutoModel(
                model=self.model_dir,
                trust_remote_code=True,
                remote_code="./model.py",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device,
            )
            print(f"语音转文字模块初始化成功，使用设备: {self.device}")
        except Exception as e:
            print(f"语音转文字模块初始化失败: {str(e)}")
            self.init_error = str(e)  # 保存错误信息
            # sys.exit(1)  # 不直接退出程序
    
    def transcribe(self, audio_path):
        """将音频转换为文本"""
        try:
            # 检查模型是否已初始化
            if self.model is None:
                error_msg = getattr(self, 'init_error', '未知错误')
                return {"success": False, "error": f"模型未初始化: {error_msg}"}
                
            if not os.path.exists(audio_path):
                return {"success": False, "error": f"音频文件不存在: {audio_path}"}
                
            # 修改模型调用方式，与 demo1.py 或 webui.py 保持一致
            res = self.model.generate(
                input=audio_path,
                cache={},
                language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            
            if not res:
                return {"success": False, "error": "转录失败，未返回结果"}
                
            text = rich_transcription_postprocess(res[0]["text"])
            return {"success": True, "text": text}
        except Exception as e:
            return {"success": False, "error": f"转录过程发生错误: {str(e)}"}