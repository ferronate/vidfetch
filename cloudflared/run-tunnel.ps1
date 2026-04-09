$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$configFile = Join-Path $scriptDir 'config.yml'

if (-not (Test-Path $configFile)) {
    throw 'Missing cloudflared/config.yml. Copy config.example.yml to config.yml and fill in the tunnel details first.'
}

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    throw 'cloudflared is not installed or is not on PATH.'
}

& cloudflared tunnel --config $configFile run
