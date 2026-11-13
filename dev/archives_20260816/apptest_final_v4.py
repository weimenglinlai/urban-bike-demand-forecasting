# -*- coding: utf-8 -*-
"""AppTest 全链路验证: 新模型(log1p target) + 归档后导入完整性"""
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(r'project\项目\基于时空数据的共享单车需求预测与调度系统')
sys.path.insert(0, os.getcwd())

from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app/app.py", default_timeout=120)
at.run()
print(f"exceptions: {len(at.exception)}")
for e in at.exception:
    print(f"  EXC: {e.message}")

# 侧边栏关键控件
print("\n=== 侧边栏控件 ===")
try:
    sels = at.sidebar.selectbox
    for s in sels:
        print(f"  selectbox: {s.label} -> {s.value}")
except Exception as e:
    print(f"  selectbox 读取失败: {e}")

try:
    nums = at.sidebar.number_input
    for n in nums:
        print(f"  number_input: {n.label} -> {n.value}")
except Exception as e:
    print(f"  number_input 读取失败: {e}")

# 主区标题/指标
print("\n=== 主区 ===")
for t in at.title:
    print(f"  title: {t.value}")
for m in at.metric:
    print(f"  metric: {m.label} = {m.value}")

print(f"\n最终 exceptions: {len(at.exception)}")
print("ALL DONE")
