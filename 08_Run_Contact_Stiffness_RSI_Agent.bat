@echo off
chcp 65001 > nul
title 求解器接触刚度参数自适应辨识与测试 (Contact Stiffness SysID)
cd /d "%~dp0"
echo ==============================================================================
echo  物理求解器接触刚度参数自适应辨识与测试系统
echo  主要功能:
echo    1. 评估接触面静载荷穿透位移与瞬态冲击响应
echo    2. 运行基于 Newton 1.7.0 与 SolverMuJoCo (CUDA) 的动力学探针
echo    3. 执行数值稳定性与阻尼比边界约束校验
echo    4. 针对不同质量与刚度对象执行参数自适应整定
echo ==============================================================================
echo.

"C:\Users\HHH\.zcode\workspace\draft-75bd8b6f-f7da-45d8-aff2-4dc217c4ed6e\.venv\Scripts\python.exe" contact_stiffness_rsi_agent.py

if exist "results\contact_stiffness_rsi_dashboard.html" (
    echo.
    echo [UI] 正在打开测试结果看板...
    start "" "results\contact_stiffness_rsi_dashboard.html"
)

echo.
echo ==============================================================================
echo  执行完毕，测试数据与看板已输出至 results\ 目录。
echo ==============================================================================
pause > nul
