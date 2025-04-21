# run_with_log.py
#!/usr/bin/env python3
"""
入口脚本：将所有 stdout 同时输出到屏幕和日志文件，不改动业务代码
用法：python run_with_log.py
"""

import sys
import os

# 确保 logs 目录存在
os.makedirs("reports", exist_ok=True)

class Tee:
    """
    将写入的数据同时写到多个 stream（如屏幕和文件）
    """
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

if __name__ == "__main__":
    # 打开日志文件
    log_f = open("reports/run_log.txt", "w", encoding="utf-8")
    # 备份原 stdout
    original_stdout = sys.stdout
    # 用 Tee 分流
    sys.stdout = Tee(original_stdout, log_f)

    try:
        # 调用你的主逻辑入口
        from main import main
        main()
    finally:
        # 还原 stdout 并关闭日志
        sys.stdout = original_stdout
        log_f.close()
