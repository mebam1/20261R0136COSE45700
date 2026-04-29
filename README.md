$existing = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess
if ($existing) { Stop-Process -Id $existing -Force }
Start-Process python -ArgumentList '-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000' -WindowStyle Hidden

# forground로 띄우기
$existing = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess
if ($existing) { Stop-Process -Id $existing -Force }
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000