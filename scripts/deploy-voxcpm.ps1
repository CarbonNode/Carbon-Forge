param([Parameter(Mandatory=$true)][string]$Ref)
$ErrorActionPreference = 'Stop'
if ($Ref -notmatch '^[a-f0-9]{40}$') { throw 'An immutable 40-character commit SHA is required' }
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$head = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -ne $Ref) { throw 'Worker checkout does not match the requested commit' }
if (& git status --porcelain --untracked-files=no) { throw 'Worker checkout has tracked edits' }
$envPath = 'C:\Programming\CarbonForge\.env'
if (!(Test-Path $envPath)) { throw 'Existing Forge environment file is missing' }
$composeArgs = @('compose', '--env-file', $envPath, '-f', 'docker-compose.voxcpm.yml')
& docker @composeArgs build voxcpm
if ($LASTEXITCODE -ne 0) { throw 'VoxCPM image build failed; running services left unchanged' }
# Public weights are downloaded without allocating any GPU memory.
& docker @composeArgs run --rm --no-deps voxcpm python -m voxcpm_service.download
if ($LASTEXITCODE -ne 0) { throw 'VoxCPM model download failed' }
& docker @composeArgs up -d --no-deps voxcpm
if ($LASTEXITCODE -ne 0) { throw 'VoxCPM worker start failed' }
$healthy = $false
for ($i=0; $i -lt 20; $i++) {
  try {
    $health = Invoke-RestMethod -TimeoutSec 5 http://localhost:5127/health
    if ($health.ok -and $health.model -eq 'openbmb/VoxCPM2') { $healthy = $true; break }
  } catch { }
  Start-Sleep -Seconds 3
}
if (!$healthy) { throw 'VoxCPM worker health check failed' }
Write-Output ('VOXCPM_DEPLOY_OK ref=' + $Ref + ' ready=' + $health.ready + ' reason=' + $health.reason)
