# Load .env into the current process, then upload dist/* to TestPyPI.
# Usage from repo root:  .\scripts\upload-testpypi.ps1
#
# .env must contain TWINE_USERNAME, TWINE_PASSWORD, TWINE_REPOSITORY_URL.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $repoRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Error ".env not found at $envFile. Copy .env.example and fill in your token."
}

Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*#') { return }
    if ($_ -match '^\s*$') { return }
    $parts = $_ -split '=', 2
    if ($parts.Count -eq 2) {
        $key = $parts[0].Trim()
        $value = $parts[1].Trim()
        Set-Item -Path "env:$key" -Value $value
    }
}

if ($env:TWINE_PASSWORD -match 'PASTE_YOUR') {
    Write-Error "TWINE_PASSWORD in .env is still the placeholder. Replace it with your real TestPyPI token."
}

$twine = Join-Path $repoRoot ".venv\Scripts\twine.exe"
$dist = Join-Path $repoRoot "dist\*"
& $twine upload $dist
