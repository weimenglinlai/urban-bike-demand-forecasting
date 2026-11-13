# -*- coding: utf-8 -*-
"""AppTest: 验证事件标注 + 色标修改后无异常"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir('project/项目/基于时空数据的共享单车需求预测与调度系统')
from streamlit.testing.v1 import AppTest

at = AppTest.from_file('app/app.py', default_timeout=60)
at.run()
print('初始异常:', [str(e.value)[:300] for e in at.exception])
at.sidebar.selectbox[1].select('🎉 活动(需求↑)').run()
print('切活动异常:', [str(e.value)[:300] for e in at.exception])
caps = [c.value for c in at.sidebar.caption]
print('活动后全网总需求:', [c for c in caps if '全网总需求' in c])
at.sidebar.selectbox[1].select('🚧 事故(需求↓)').run()
print('切事故异常:', [str(e.value)[:300] for e in at.exception])
caps2 = [c.value for c in at.sidebar.caption]
print('事故后全网总需求:', [c for c in caps2 if '全网总需求' in c])
print('完成')
