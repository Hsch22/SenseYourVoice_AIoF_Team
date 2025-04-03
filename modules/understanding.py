#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright (c) 2023. All Rights Reserved.

import requests

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