#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright (c) 2023. All Rights Reserved.

import os
import sys
import argparse
import torch
from pathlib import Path

# 导入模块化组件
from modules import VoiceToTextModule, UnderstandingModule, SpecializedTaskModule

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