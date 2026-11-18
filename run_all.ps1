# 全部实验按顺序跑一遍（CPU 上约 2.5–3 小时）。各步结果写在 reports/ 和 runs/ 下，日志在 logs/。
# 已经有结果的步骤会跳过，中途断了重新运行即可。
# PowerShell 5 会把原生命令写到 stderr 的警告当成错误，所以不用 Stop，改为检查每步的产出文件
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null

function Step($name, $done, [scriptblock]$cmd) {
    if (Test-Path $done) { Write-Host "跳过 $name（已有结果）"; return }
    Write-Host "== $name  $(Get-Date -Format HH:mm:ss)"
    & $cmd *>&1 | Where-Object { $_ -notmatch "Loading weights|it/s\]" } | Tee-Object -FilePath "logs\$name.log"
    if (-not (Test-Path $done)) { throw "$name 没有产出 $done" }
}

Step "train-n200" "runs\lora-n200\adapter" { uv run python -m finlora.train --name lora-n200 --n-train 200 --n-dev 60 --eval-every 4 }
Step "eval-lora-n200" "reports\metrics\lora-0.5b-n200.json" { uv run python -m finlora.predict --name lora-0.5b-n200 --adapter runs\lora-n200\adapter }
Step "eval-base-zeroshot" "reports\metrics\base-0.5b-zeroshot.json" { uv run python -m finlora.predict --name base-0.5b-zeroshot }
Step "eval-base-fewshot" "reports\metrics\base-0.5b-fewshot3.json" { uv run python -m finlora.predict --name base-0.5b-fewshot3 --shots 3 }
Step "train-n800" "runs\lora-n800\adapter" { uv run python -m finlora.train --name lora-n800 --n-train 800 --n-dev 60 --eval-every 10 }
Step "eval-lora-n800" "reports\metrics\lora-0.5b-n800.json" { uv run python -m finlora.predict --name lora-0.5b-n800 --adapter runs\lora-n800\adapter }
Step "eval-ollama-1.5b" "reports\metrics\ollama-1.5b-fewshot3.json" { uv run python -m finlora.predict --backend ollama --model qwen2.5:1.5b --name ollama-1.5b-fewshot3 --shots 3 }
Step "eval-ollama-3b-zero" "reports\metrics\ollama-3b-zeroshot.json" { uv run python -m finlora.predict --backend ollama --model qwen2.5:3b --name ollama-3b-zeroshot }
Step "eval-ollama-3b" "reports\metrics\ollama-3b-fewshot3.json" { uv run python -m finlora.predict --backend ollama --model qwen2.5:3b --name ollama-3b-fewshot3 --shots 3 }
Write-Host "== 全部完成 $(Get-Date -Format HH:mm:ss)"
