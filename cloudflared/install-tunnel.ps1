param(
    [switch]$SkipLogin
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$configExample = Join-Path $scriptDir 'config.example.yml'
$configFile = Join-Path $scriptDir 'config.yml'

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    throw 'cloudflared is not installed or is not on PATH. Install it first, then rerun this script.'
}

if (-not (Test-Path $configFile)) {
    Copy-Item $configExample $configFile
    Write-Host "Created $configFile from the template. Fill in the tunnel UUID, credentials path, and hostname."
}

if (-not $SkipLogin) {
    Write-Host 'Opening Cloudflare login in the browser...'
    & cloudflared tunnel login
}

Write-Host ''
Write-Host 'Next steps:'
Write-Host '1. Create a tunnel with: cloudflared tunnel create <tunnel-name>'
Write-Host '2. Update cloudflared/config.yml with the generated UUID and credentials file.'
Write-Host '3. Run cloudflared/run-tunnel.ps1 after the backend is listening on localhost:8000.'
