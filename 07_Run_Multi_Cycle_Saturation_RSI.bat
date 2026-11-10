@echo off
chcp 65001 > nul
title [Embodied Robotics Sim] 07 - 多轮自适应自饱和深度 RSI 引擎 (教师门禁 0 RSI 停机裁决)
cd /d "%~dp0"
echo ==============================================================================
echo  [Embodied Robotics Sim 面试演示] 07 - 多轮自适应自饱和深度 RSI 引擎
echo  终极准则: 持续多轮自主协同进化，直至独立教师模型判定连续 3 轮 0 RSI 增益方可终止！
echo  协同维度: 碗夹爪 (g_bowl) + 杯夹爪 (g_cup) + 工业节拍加速 (Speed Scale 0.85x)
echo  门禁审计: 不可变契约硬锁 + 留出测试集 (Held-out Seeds 4~7) 跨域泛化双盲认证
echo ==============================================================================
echo.
"C:\Users\HHH\.zcode\workspace\draft-75bd8b6f-f7da-45d8-aff2-4dc217c4ed6e\.venv\Scripts\python.exe" rsi_multi_cycle_saturation_engine.py
echo.
echo 执行完毕，教师多轮 0 RSI 饱和认证报告已导出至 results\rsi_deep_autonomous_saturation_report.json。
echo 请按任意键退出...
pause > nul
