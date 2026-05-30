param(
    [string]$PythonVersion = "3.12",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\\python.exe"
$LocalDiffusersPath = Join-Path $ProjectRoot "cog_diffuser\\diffusers"

Write-Host "Creating virtual environment with Python $PythonVersion ..."
uv venv --python $PythonVersion $VenvPath

Write-Host "Upgrading pip ..."
uv pip install --python $PythonExe --upgrade pip setuptools wheel

Write-Host "Installing PyTorch from $TorchIndexUrl ..."
uv pip install --python $PythonExe torch torchvision torchaudio --index-url $TorchIndexUrl

Write-Host "Installing project requirements ..."
uv pip install --python $PythonExe -r (Join-Path $ProjectRoot "requirements.txt")

if (Test-Path $LocalDiffusersPath) {
    Write-Host "Installing local diffusers source in editable mode ..."
    uv pip install --python $PythonExe -e $LocalDiffusersPath
} else {
    Write-Host "Local diffusers source not found at $LocalDiffusersPath"
}

Write-Host "Environment setup complete."
Write-Host "Activate with: .\\.venv\\Scripts\\Activate.ps1"
Write-Host "Verify with: python scripts/check_env.py"
