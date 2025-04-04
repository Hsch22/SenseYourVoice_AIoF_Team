#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright (c) 2023. All Rights Reserved.

import requests
import logging
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('UnderstandingModule')

class UnderstandingModule:
    def __init__(self, api_key=None, api_url=None, model="gpt-3.5-turbo"):
        """理解与分析模块，可以连接到外部LLM API"""
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        logger.info("理解模块初始化完成")
        
    def analyze(self, text, instruction=""):
        """分析文本内容并返回结果"""
        # 这里可以实现连接到外部LLM API的逻辑
        # 示例实现，实际使用时需要替换为真实的API调用
        try:
            # 如果没有配置API，返回模拟响应
            if not self.api_url or not self.api_key:
                logger.warning("未配置API，返回模拟响应")
                return {
                    "success": True, 
                    "response": f"模拟分析结果: 已收到文本，长度为{len(text)}字符。请配置真实的LLM API以获取实际分析结果。",
                    "needs_specialized_task": False
                }
            
            # 构建系统提示词
            system_prompt = "你是一个专业的语音内容分析助手。你的任务是分析用户提供的语音转文字内容，并根据用户的指令提供相应的分析结果。"
            
            # 如果有指令，则添加到系统提示中
            if instruction:
                system_prompt += f"\n请特别注意用户的指令: {instruction}"
            
            # 实际API调用
            logger.info(f"准备调用API分析文本，文本长度: {len(text)}字符")
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            }
            
            # 添加温度参数，使结果更稳定
            payload["temperature"] = 0.7
            
            logger.debug(f"API请求: {json.dumps(payload, ensure_ascii=False)}")
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                logger.info("API调用成功，获取到分析结果")
                
                # 检查是否需要专业任务处理
                needs_specialized = self._check_if_needs_specialized_task(response_text)
                if needs_specialized:
                    logger.info("检测到需要专业任务处理")
                
                return {
                    "success": True,
                    "response": response_text,
                    "needs_specialized_task": needs_specialized
                }
            else:
                error_msg = f"API调用失败: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except requests.exceptions.Timeout:
            error_msg = "API请求超时"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求异常: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"分析过程发生错误: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _check_if_needs_specialized_task(self, response):
        """检查是否需要专业任务处理"""
        # 检查响应中是否包含特定关键词，表明需要专业任务处理
        keywords = [
            "代码", "编程", "程序", "算法", "函数", "变量", "类", "对象", 
            "数学问题", "计算", "方程", "公式", "数值", "统计", 
            "搜索", "查询", "检索", "查找", "数据库", 
            "专业分析", "深度解析", "技术细节"
        ]
        
        # 检查是否包含任何关键词
        for keyword in keywords:
            if keyword in response:
                logger.debug(f"检测到专业任务关键词: {keyword}")
                return True
                
        return False