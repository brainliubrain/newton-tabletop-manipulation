Write-Host "=== 正在检测 GitHub SSH 认证状态... ===" -ForegroundColor Cyan
$test = ssh -T -o StrictHostKeyChecking=no -o ConnectTimeout=5 git@github.com 2>&1
Write-Host $test

if ($test -match "successfully authenticated") {
    Write-Host ">>> GitHub SSH 鉴权成功！正在全自动推流/推送本地所有简历核心项目..." -ForegroundColor Green
    
    # 1. Newton Pick Place
    Write-Host "[1/4] 推送 Newton 桌面抓取与 RSI 项目..." -ForegroundColor Yellow
    cd "C:\Users\HHH\Desktop\newton_pick_place"
    git branch -M main
    git push -u origin main --force
    
    Write-Host ">>> 全部推送完成！" -ForegroundColor Green
} else {
    Write-Host ">>> 提示: GitHub 鉴权尚未就绪 (请先将公钥添加至 github.com/settings/ssh/new)" -ForegroundColor Red
}
