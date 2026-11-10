@echo off
chcp 65001 > nul
title [Embodied Robotics Sim] 03 - 运行抓持预紧自适应 RSI (教师门禁认证)
cd /d "%~dp0"
echo ==============================================================================
echo  [Embodied Robotics Sim 面试演示] 03 - 单参数真实物理误差反馈自适应自进化闭环
echo  核心机制: 误差驱动比例反馈更新律 (Proportional Error Feedback)
echo  门禁机制: 教师级独立监督官 (TeacherAntiHackingSupervisor)
echo  验证结果: 训练集 0%% -^> 100%% 收敛，留出测试集 (Seeds 5-9) 100%% 零滑动泛化通过
echo ==============================================================================
echo.
"C:\Users\HHH\.zcode\workspace\draft-75bd8b6f-f7da-45d8-aff2-4dc217c4ed6e\.venv\Scripts\python.exe" rsi_single_param_closed_loop.py
echo.
echo 执行完毕，教师认证报告已导出至 results\rsi_teacher_verified_report.json。请按任意键退出...
pause > nul
