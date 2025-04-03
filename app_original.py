#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright (c) 2023. All Rights Reserved.

import os
import sys
import argparse
import torch
import requests
from pathlib import Path

# 语音转文字模块
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
            self.model = AutoModel(
                model=self.model_dir,
                trust_remote_code=True,
                remote_code="./model.py",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device,
            )
            print(f"语音转文字模块初始化成功，使用设备: {self.device}")
            # 检查模型是否正确初始化
            if self.model is None:
                raise Exception("模型初始化后为None")
        except Exception as e:
            print(f"语音转文字模块初始化失败: {str(e)}")
            self.init_error = str(e)  # 保存错误信息而不是直接退出
            # sys.exit(1)  # 不立即退出
    
    def transcribe(self, audio_path):
        """将音频转换为文本"""
        try:
            # 检查模型是否已初始化
            if self.model is None:
                error_msg = getattr(self, 'init_error', '未知错误')
                return {"success": False, "error": f"模型未初始化: {error_msg}"}
                
            if not os.path.exists(audio_path):
                return {"success": False, "error": f"音频文件不存在: {audio_path}"}
            
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


class UnderstandingModule:
    def __init__(self, api_key=None, api_url=None):
        """理解与分析模块，可以连接到外部LLM API"""
        self.api_key = api_key
        self.api_url = api_url
        
    def analyze(self, text, instruction=""):
        """分析文本内容并返回结果"""
        # 这里可以实现连接到外部LLM API的逻辑
        # 示例实现，实际使用时需要替换为真实的API调用
        try:
            # 如果没有配置API，返回模拟响应
            if not self.api_url:
                return {
                    "success": True, 
                    "response": f"模拟分析结果: 已收到文本，长度为{len(text)}字符。请配置真实的LLM API以获取实际分析结果。",
                    "needs_specialized_task": False
                }
            
            # 实际API调用示例
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": "gpt-3.5-turbo",  # 或其他模型
                "messages": [
                    {"role": "system", "content": "你是一个专业的语音内容分析助手。"},
                    {"role": "user", "content": f"指令: {instruction}\n\n内容: {text}"}
                ]
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result["choices"][0]["message"]["content"],
                    "needs_specialized_task": self._check_if_needs_specialized_task(result["choices"][0]["message"]["content"])
                }
            else:
                return {"success": False, "error": f"API调用失败: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"分析过程发生错误: {str(e)}"}
    
    def _check_if_needs_specialized_task(self, response):
        """检查是否需要专业任务处理"""
        # 简单示例：检查响应中是否包含特定关键词
        keywords = ["代码", "编程", "数学问题", "计算", "搜索", "查询"]
        return any(keyword in response for keyword in keywords)


class SpecializedTaskModule:
    def __init__(self, api_key=None, api_url=None):
        """专业任务处理模块，用于处理代码、数学问题等"""
        self.api_key = api_key
        self.api_url = api_url
        
    def process_task(self, task_type, content):
        """处理特定类型的任务"""
        # 示例实现，实际使用时需要替换为真实的API调用
        try:
            if not self.api_url:
                return {
                    "success": True, 
                    "result": f"模拟专业任务处理结果: 已收到{task_type}任务，内容长度为{len(content)}字符。请配置真实的专业LLM API以获取实际处理结果。"
                }
            
            # 实际API调用示例
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": "gpt-4",  # 或其他适合专业任务的模型
                "messages": [
                    {"role": "system", "content": f"你是一个专业的{task_type}处理助手。"},
                    {"role": "user", "content": content}
                ]
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "result": result["choices"][0]["message"]["content"]
                }
            else:
                return {"success": False, "error": f"API调用失败: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"专业任务处理过程发生错误: {str(e)}"}


class SenseYourVoiceApp:
    def __init__(self, config=None):
        """主应用类，整合三个模块"""
        self.config = config or {}
        
        # 初始化三个模块
        self.voice_to_text = VoiceToTextModule(
            model_dir=self.config.get("model_dir", "iic/SenseVoiceSmall"),
            device=self.config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        )
        
        self.understanding = UnderstandingModule(
            api_key=self.config.get("understanding_api_key"),
            api_url=self.config.get("understanding_api_url")
        )
        
        self.specialized_task = SpecializedTaskModule(
            api_key=self.config.get("specialized_api_key"),
            api_url=self.config.get("specialized_api_url")
        )
        
        print("SenseYourVoice应用初始化完成")
    
    def process(self, audio_path, instruction=""):
        """处理音频文件的完整流程"""
        # 步骤1: 语音转文字
        print(f"正在处理音频文件: {audio_path}")
        transcription_result = self.voice_to_text.transcribe(audio_path)
        
        if not transcription_result["success"]:
            return {"success": False, "error": transcription_result["error"]}
        
        text = transcription_result["text"]
        print(f"语音转文字完成，文本长度: {len(text)}字符")
        
        # 步骤2: 理解与分析
        understanding_result = self.understanding.analyze(text, instruction)
        
        if not understanding_result["success"]:
            return {"success": False, "error": understanding_result["error"]}
        
        response = understanding_result["response"]
        needs_specialized_task = understanding_result.get("needs_specialized_task", False)
        
        # 步骤3: 如果需要，进行专业任务处理
        if needs_specialized_task:
            print("检测到需要专业任务处理")
            # 这里可以根据理解模块的输出确定任务类型
            task_type = self._determine_task_type(response)
            specialized_result = self.specialized_task.process_task(task_type, response)
            
            if not specialized_result["success"]:
                return {"success": False, "error": specialized_result["error"]}
            
            final_result = {
                "success": True,
                "transcription": text,
                "understanding": response,
                "specialized_result": specialized_result["result"]
            }
        else:
            final_result = {
                "success": True,
                "transcription": text,
                "understanding": response
            }
        
        return final_result
    
    def _determine_task_type(self, text):
        """根据文本内容确定专业任务类型"""
        if "代码" in text or "编程" in text:
            return "代码处理"
        elif "数学" in text or "计算" in text:
            return "数学问题"
        elif "搜索" in text or "查询" in text:
            return "网络搜索"
        else:
            return "通用任务"


def parse_args():
    parser = argparse.ArgumentParser(description="SenseYourVoice - 语音理解与处理应用")
    parser.add_argument("--audio", type=str, required=True, help="音频文件路径")
    parser.add_argument("--instruction", type=str, default="", help="处理指令")
    parser.add_argument("--model_dir", type=str, default="iic/SenseVoiceSmall", help="语音模型目录")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--understanding_api_key", type=str, help="理解模块API密钥")
    parser.add_argument("--understanding_api_url", type=str, help="理解模块API地址")
    parser.add_argument("--specialized_api_key", type=str, help="专业任务模块API密钥")
    parser.add_argument("--specialized_api_url", type=str, help="专业任务模块API地址")
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = {
        "model_dir": args.model_dir,
        "device": args.device,
        "understanding_api_key": args.understanding_api_key,
        "understanding_api_url": args.understanding_api_url,
        "specialized_api_key": args.specialized_api_key,
        "specialized_api_url": args.specialized_api_url
    }
    
    app = SenseYourVoiceApp(config)
    result = app.process(args.audio, args.instruction)
    
    if result["success"]:
        print("\n===== 处理结果 =====")
        print(f"\n语音转文字结果:\n{result['transcription']}")
        print(f"\n理解分析结果:\n{result['understanding']}")
        if "specialized_result" in result:
            print(f"\n专业任务处理结果:\n{result['specialized_result']}")
    else:
        print(f"\n处理失败: {result['error']}")


if __name__ == "__main__":
    main()