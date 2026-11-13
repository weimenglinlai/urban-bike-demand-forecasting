# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""AppTest: 驱动事件扰动 selectbox, 验证数值变化 (科学验证事件链路)"""
import sys, os
sys.path.insert(0, "project/项目/基于时空数据的共享单车需求预测与调度系统/src")
sys.path.insert(0, "project/项目/基于时空数据的共享单车需求预测与调度系统")
os.chdir("project/项目/基于时空数据的共享单车需求预测与调度系统")

from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app/app.py", default_timeout=60)
at.run()

def get_total():
    """从侧边栏监控读全网总需求"""
    for c in at.sidebar:
        pass
    # 遍历 caption
    caps = [c.value for c in at.sidebar.caption]
    return caps

print("=== 初始 (无事件) ===")
print("captions:", get_total())

# 切换事件扰动 selectbox
# 找 selectbox index
sb = at.sidebar.selectbox
print(f"\nselectbox 数量: {len(sb)}")
for i, s in enumerate(sb):
    print(f"  [{i}] label={s.label!r} value={s.value!r}")

# 事件扰动 selectbox 应该是 index 1 (天气0, 事件1, 模型2, 节假日3)
evt_sb = sb[1]
print(f"\n事件扰动当前值: {evt_sb.value!r}")
evt_sb.select("🎉 活动(需求↑)").run()
print(f"切换后值: {evt_sb.value!r}")

# 检查事件强度 slider 出现
sliders = at.sidebar.slider
print(f"\nslider 数量: {len(sliders)}")
for i, s in enumerate(sliders):
    print(f"  [{i}] label={s.label!r} value={s.value!r}")

print("\n=== 切换活动后 ===")
print("captions:", get_total())

# 切到事故
try:
    at.sidebar.selectbox[1].select("🚧 事故(需求↓)").run()
    print(f"\n切换后值: {at.sidebar.selectbox[1].value!r}")
    print("=== 切换事故后 ===")
    print("captions:", get_total())
except Exception as e:
    print("切换事故失败:", e)

print("\n=== 异常检查 ===")
print("exceptions:", [str(e.value)[:200] for e in at.exception])
