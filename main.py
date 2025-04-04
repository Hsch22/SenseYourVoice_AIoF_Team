#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright (c) 2023. All Rights Reserved.

import os
import sys
import argparse
import torch
import gradio as gr

# 导入配置和主应用
from config import load_config
from app_new import SenseYourVoiceApp

# 全局应用实例
sense_app = None

def initialize_app(model_dir, device, understanding_api_key, understanding_api_url, specialized_api_key, specialized_api_url):
    """初始化应用实例"""
    global sense_app
    
    # 从用户输入创建配置字典
    user_config = {
        "model_dir": model_dir,
        "device": device,
        "understanding_api_key": understanding_api_key,
        "understanding_api_url": understanding_api_url,
        "specialized_api_key": specialized_api_key,
        "specialized_api_url": specialized_api_url
    }
    
    # 使用load_config函数加载配置，合并用户配置和默认配置
    config = load_config(user_config)
    
    try:
        sense_app = SenseYourVoiceApp(config)
        return "应用初始化成功！"
    except Exception as e:
        return f"应用初始化失败: {str(e)}"

def process_audio(audio_file, instruction):
    """处理上传的音频文件"""
    global sense_app
    
    if sense_app is None:
        return None, None, None, "应用尚未初始化，请先初始化应用。"
    
    if audio_file is None:
        return None, None, None, "请上传音频文件。"
    
    try:
        # 处理音频文件
        result = sense_app.process(audio_file, instruction)
        
        if not result["success"]:
            return None, None, None, result["error"]
        
        # 构建返回结果
        transcription = result["transcription"]
        understanding = result["understanding"]
        specialized_result = result.get("specialized_result", None)
        
        return transcription, understanding, specialized_result, None
    except Exception as e:
        return None, None, None, f"处理过程发生错误: {str(e)}"

def main():
    # 加载默认配置
    default_config = load_config()
    
    # 解析命令行参数，使用config中的默认值
    parser = argparse.ArgumentParser(description="SenseYourVoice - Gradio WebUI")
    parser.add_argument("--model_dir", type=str, default=default_config["model_dir"], help="语音模型目录")
    parser.add_argument("--device", type=str, default=default_config["device"], help="设备")
    parser.add_argument("--understanding_api_key", type=str, default=default_config["understanding_api_key"], help="理解模块API密钥")
    parser.add_argument("--understanding_api_url", type=str, default=default_config["understanding_api_url"], help="理解模块API地址")
    parser.add_argument("--specialized_api_key", type=str, default=default_config["specialized_api_key"], help="专业任务模块API密钥")
    parser.add_argument("--specialized_api_url", type=str, default=default_config["specialized_api_url"], help="专业任务模块API地址")
    parser.add_argument("--auto_init", action="store_true", default=default_config["auto_init"], help="自动初始化应用")
    parser.add_argument("--share", action="store_true", default=default_config["share"], help="创建公共链接分享界面")
    parser.add_argument("--port", type=int, default=default_config["port"], help="服务端口")
    args = parser.parse_args()
    
    # 创建Gradio界面
    with gr.Blocks(title="SenseYourVoice - 语音理解与处理") as demo:
        gr.Markdown("""
        # SenseYourVoice - 语音理解与处理
        
        上传音频文件，获取语音转文字结果和智能分析。
        """)
        
        # 应用设置部分
        with gr.Tab("应用设置"):
            model_dir = gr.Textbox(label="语音模型目录", value=args.model_dir)
            device = gr.Dropdown(
                label="设备", 
                choices=["cuda:0", "cpu"], 
                value=args.device
            )
            understanding_api_key = gr.Textbox(label="理解模块API密钥", value=args.understanding_api_key or "")
            understanding_api_url = gr.Textbox(label="理解模块API地址", value=args.understanding_api_url or "")
            specialized_api_key = gr.Textbox(label="专业任务模块API密钥", value=args.specialized_api_key or "")
            specialized_api_url = gr.Textbox(label="专业任务模块API地址", value=args.specialized_api_url or "")
            
            init_btn = gr.Button("初始化应用")
            init_output = gr.Textbox(label="初始化状态")
            
            init_btn.click(
                fn=initialize_app,
                inputs=[model_dir, device, understanding_api_key, understanding_api_url, specialized_api_key, specialized_api_url],
                outputs=init_output
            )
        
        # 语音处理部分
        with gr.Tab("语音处理"):
            audio_input = gr.Audio(label="上传音频文件", type="filepath")
            instruction = gr.Textbox(label="处理指令（可选）", placeholder="输入特定的处理指令，例如：'总结这段语音的主要内容'或'分析这段语音中的情感'等")
            process_btn = gr.Button("开始处理")
            
            with gr.Row():
                transcription_output = gr.Textbox(label="语音转文字结果")
                understanding_output = gr.Textbox(label="理解分析结果")
            
            specialized_output = gr.Textbox(label="专业任务处理结果", visible=False)
            error_output = gr.Textbox(label="错误信息", visible=False)
            
            def process_and_update(audio_file, instruction):
                # 先调用 process_audio 处理音频
                transcription, understanding, specialized, error = process_audio(audio_file, instruction)
                
                # 然后处理输出格式
                if error:
                    return (
                        "", 
                        "", 
                        gr.update(value="", visible=False),
                        gr.update(value=error, visible=True)
                    )
                
                return (
                    transcription, 
                    understanding, 
                    gr.update(value=specialized if specialized else "", visible=specialized is not None),
                    gr.update(value="", visible=False)
                )
            
            process_btn.click(
                fn=process_and_update,
                inputs=[audio_input, instruction],
                outputs=[transcription_output, understanding_output, specialized_output, error_output]
            )
        
        gr.Markdown("""
        ### 使用说明
        1. 首先在"应用设置"标签页中配置应用参数并初始化应用
        2. 然后在"语音处理"标签页中上传音频文件并开始处理
        3. 如果需要特定的分析指令，可以在处理指令文本框中输入
        """)
    
    # 如果设置了自动初始化，则在启动前初始化应用
    if args.auto_init:
        init_result = initialize_app(
            args.model_dir,
            args.device,
            args.understanding_api_key,
            args.understanding_api_url,
            args.specialized_api_key,
            args.specialized_api_url
        )
        print(f"自动初始化结果: {init_result}")
    
    # 启动Gradio界面
    demo.launch(share=args.share, server_port=args.port)

if __name__ == "__main__":
    main()
    