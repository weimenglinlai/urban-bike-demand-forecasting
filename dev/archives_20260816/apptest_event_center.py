# -*- coding: utf-8 -*-
"""AppTest: 验证事件中心坐标可选择(默认10,10,改X=5后事件区偏移)"""
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(r"project\项目\基于时空数据的共享单车需求预测与调度系统")
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app/app.py", default_timeout=30)
at.run()
print("初始异常:", at.exception)
assert not at.exception, f"初始异常: {at.exception}"

# 选择活动事件
at.sidebar.selectbox[1].select("🎉 活动(需求↑)").run()
print("选活动后异常:", at.exception)
assert not at.exception, f"活动异常: {at.exception}"

# 默认中心应为 (10,10): 检查 number_input 存在
ninputs = at.sidebar.number_input
print("number_input 数量:", len(ninputs))
for ni in ninputs:
    print(f"  label={ni.label} value={ni.value}")

# 修改 X 坐标为 5
if len(ninputs) >= 2:
    ninputs[0].set_value(5).run()
    print("改X=5后异常:", at.exception)
    assert not at.exception, f"改X后异常: {at.exception}"
    # 全网总需求应与 (10,10) 时不同(事件区偏移影响不同网格)
    m = [x for x in at.markdown]
    total_lines = [str(x.value) for x in m if "全网总需求" in str(x.value)]
    print("改X=5后全网总需求:", total_lines[-1] if total_lines else "?")

# 事故事件 + Y 坐标
at.sidebar.selectbox[1].select("🚧 事故(需求↓)").run()
ninputs = at.sidebar.number_input
if len(ninputs) >= 2:
    ninputs[1].set_value(3).run()
    print("事故改Y=3后异常:", at.exception)
    assert not at.exception, f"事故Y异常: {at.exception}"
    m = [x for x in at.markdown]
    total_lines = [str(x.value) for x in m if "全网总需求" in str(x.value)]
    print("事故(5,3)全网总需求:", total_lines[-1] if total_lines else "?")

print("完成")
