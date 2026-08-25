# fix_electron.ps1 - repairs a broken Electron binary install and launches the app
#
# Use this when `npm run start` fails with:
#   "Electron failed to install correctly, please delete node_modules/electron and try installing again"
# This usually means the Electron binary download was interrupted/blocked and
# node_modules\electron\dist is incomplete (e.g. only a "locales" folder, no electron.exe).
#
# Usage (from the electron-app folder):
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force   # first time only
#   .\fix_electron.ps1

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

Write-Host "Cleaning old electron dist..." -ForegroundColor Cyan
if (Test-Path "node_modules\electron\dist") {
    Remove-Item -Recurse -Force "node_modules\electron\dist"
}

Write-Host "Setting Electron mirror..." -ForegroundColor Cyan
$env:ELECTRON_MIRROR = "https://npmmirror.com/mirrors/electron/"

Write-Host "Reinstalling Electron binary..." -ForegroundColor Cyan
node node_modules\electron\install.js

Write-Host "Verifying dist folder..." -ForegroundColor Cyan
$exe = "node_modules\electron\dist\electron.exe"
if (-not (Test-Path $exe)) {
    Write-Host "electron.exe still missing - download likely blocked. Aborting." -ForegroundColor Red
    exit 1
}

Write-Host "Electron binary looks good. Starting app..." -ForegroundColor Green
npm run start
