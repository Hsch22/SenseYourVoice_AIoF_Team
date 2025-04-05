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

def process_audio(audio_file, instruction, chat_history):
    """处理上传的音频文件并更新对话历史"""
    global sense_app
    
    if sense_app is None:
        return chat_history, None, "应用尚未初始化，请先初始化应用。"
    
    if audio_file is None:
        return chat_history, None, "请上传音频文件。"
    
    try:
        # 构建完整的对话历史上下文
        context = ""
        if chat_history:
            for user_msg, bot_msg in chat_history:
                if user_msg and bot_msg:
                    context += f"用户: {user_msg}\n助手: {bot_msg}\n"
        
        # 处理音频文件，传入对话历史上下文
        result = sense_app.process(audio_file, instruction, context)
        
        if not result["success"]:
            return chat_history, None, result["error"]
        
        # 构建返回结果
        transcription = result["transcription"]
        understanding = result["understanding"]
        specialized_result = result.get("specialized_result", None)
        
        # 更新对话历史
        new_chat_history = list(chat_history) if chat_history else []
        new_chat_history.append((transcription, understanding))
        
        return new_chat_history, specialized_result, None
    except Exception as e:
        return chat_history, None, f"处理过程发生错误: {str(e)}"

def process_text(text_input, instruction, chat_history):
    """处理用户输入的文本并更新对话历史，跳过语音转文字步骤"""
    global sense_app
    
    if sense_app is None:
        return chat_history, None, "应用尚未初始化，请先初始化应用。"
    
    if not text_input or text_input.strip() == "":
        return chat_history, None, "请输入文本内容。"
    
    try:
        # 构建完整的对话历史上下文
        context = ""
        if chat_history:
            for user_msg, bot_msg in chat_history:
                if user_msg and bot_msg:
                    context += f"用户: {user_msg}\n助手: {bot_msg}\n"
        
        # 直接调用理解模块分析文本，跳过语音转文字步骤
        understanding_result = sense_app.understanding.analyze(text_input, instruction, context)
        
        if not understanding_result["success"]:
            return chat_history, None, understanding_result["error"]
        
        response = understanding_result["response"]
        needs_specialized_task = understanding_result.get("needs_specialized_task", False)
        
        # 如果需要，进行专业任务处理
        specialized_result = None
        if needs_specialized_task:
            task_type = sense_app._determine_task_type(response)
            specialized_task_result = sense_app.specialized_task.process_task(task_type, response)
            
            if specialized_task_result["success"]:
                specialized_result = specialized_task_result["result"]
        
        # 更新对话历史
        new_chat_history = list(chat_history) if chat_history else []
        new_chat_history.append((text_input, response))
        
        return new_chat_history, specialized_result, None
    except Exception as e:
        return chat_history, None, f"处理过程发生错误: {str(e)}"

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
            # 添加State组件存储对话历史
            chat_history = gr.State([])
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 音频输入")
                    audio_input = gr.Audio(label="上传音频文件", type="filepath")
                    process_audio_btn = gr.Button("处理音频")
                
                with gr.Column():
                    gr.Markdown("### 文本输入")
                    text_input = gr.Textbox(label="输入文本", placeholder="直接输入文本内容，无需语音转文字")
                    process_text_btn = gr.Button("继续对话")
            
            instruction = gr.Textbox(label="处理指令（可选）", placeholder="输入特定的处理指令，例如：'总结这段语音的主要内容'或'分析这段语音中的情感'等")
            
            # 添加Chatbot组件显示对话历史
            chatbot = gr.Chatbot(label="对话历史", height=400)
            
            clear_btn = gr.Button("清除对话历史")
            
            specialized_output = gr.Textbox(label="专业任务处理结果", visible=False)
            error_output = gr.Textbox(label="错误信息", visible=False)
            
            def process_and_update(audio_file, instruction, history):
                # 调用 process_audio 处理音频，传入对话历史
                new_history, specialized, error = process_audio(audio_file, instruction, history)
                
                # 处理输出格式
                if error:
                    return (
                        history,  # 保持原有对话历史不变
                        history,  # Chatbot显示不变
                        gr.update(value="", visible=False),
                        gr.update(value=error, visible=True)
                    )
                
                return (
                    new_history,  # 更新State中的对话历史
                    new_history,  # 更新Chatbot显示
                    gr.update(value=specialized if specialized else "", visible=specialized is not None),
                    gr.update(value="", visible=False)
                )
            
            def clear_chat_history():
                return [], [], gr.update(value="", visible=False), gr.update(value="", visible=False)
            
            # 处理音频按钮事件
            process_audio_btn.click(
                fn=process_and_update,
                inputs=[audio_input, instruction, chat_history],
                outputs=[chat_history, chatbot, specialized_output, error_output]
            )
            
            # 处理文本按钮事件
            def process_text_and_update(text, instruction, history):
                # 调用 process_text 处理文本，传入对话历史
                new_history, specialized, error = process_text(text, instruction, history)
                
                # 处理输出格式
                if error:
                    return (
                        history,  # 保持原有对话历史不变
                        history,  # Chatbot显示不变
                        gr.update(value="", visible=False),
                        gr.update(value=error, visible=True)
                    )
                
                return (
                    new_history,  # 更新State中的对话历史
                    new_history,  # 更新Chatbot显示
                    gr.update(value=specialized if specialized else "", visible=specialized is not None),
                    gr.update(value="", visible=False)
                )
            
            process_text_btn.click(
                fn=process_text_and_update,
                inputs=[text_input, instruction, chat_history],
                outputs=[chat_history, chatbot, specialized_output, error_output]
            )
            
            clear_btn.click(
                fn=clear_chat_history,
                inputs=[],
                outputs=[chat_history, chatbot, specialized_output, error_output]
            )
        
        gr.Markdown("""
        ### 使用说明
        1. 首先在"应用设置"标签页中配置应用参数并初始化应用
        2. 在"语音处理"标签页中有两种交互方式：
           - **处理音频**：上传音频文件，点击"处理音频"按钮进行语音转文字和分析
           - **继续对话**：直接在文本输入框中输入内容，点击"继续对话"按钮跳过语音转文字步骤
        3. 如果需要特定的分析指令，可以在处理指令文本框中输入
        4. 使用"清除对话历史"按钮可以开始新的对话
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
    