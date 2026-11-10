@echo off
chcp 65001 > nul
title [Embodied Robotics Sim] 02 - 运行 Solver 接触刚度自适应 RSI (秒级收敛)
cd /d "%~dp0"
echo ==============================================================================
echo  [Embodied Robotics Sim 面试演示] 02 - 求解器系统辨识 (Solver SysID) 自适应 RSI 闭环
echo  重点考察: 仿真 Infra / Solver 接触刚度 (ke, kd) 与穿透误差 (Penetration) 系统辨识
echo  收敛过程: Round 0 (8.59mm 严重穿透) -^> Round 1 (0.22mm 亚毫米收敛)
echo  执行时间: 约 1.5 秒 (秒级极速，面试现场可即时实机演示)
echo ==============================================================================
echo.
"C:\Users\HHH\.zcode\workspace\draft-75bd8b6f-f7da-45d8-aff2-4dc217c4ed6e\.venv\Scripts\python.exe" rsi_solver_sysid_live.py
echo.
echo 执行完毕，报告已导出至 results\rsi_solver_sysid_report.json。请按任意键退出...
pause > nul
