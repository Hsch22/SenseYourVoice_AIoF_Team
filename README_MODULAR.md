# SenseYourVoice 模块化版本

这是SenseYourVoice应用的模块化版本，将核心功能分离到独立的模块中，提高了代码的可维护性和可扩展性。

## 项目结构

```
SenseYourVoice/
├── app.py                # 主应用入口
├── webui_app.py          # 基于Flask的Web界面
├── modules/              # 模块化组件
│   ├── __init__.py       # 包初始化文件
│   ├── voice_to_text.py  # 语音转文字模块
│   ├── understanding.py  # 理解与分析模块
│   └── specialized_task.py # 专业任务处理模块
└── ... (其