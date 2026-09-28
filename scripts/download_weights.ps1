# Resumable download of the GPT-1 files into weights\openai-gpt\ (PowerShell, Windows 10/11).
# Uses the curl.exe that ships with Windows; "-C -" resumes a partial file after any drop.
# The code picks the local folder up automatically (see src\paths.py).
#   powershell -ExecutionPolicy Bypass -File scripts\download_weights.ps1
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$dir = "weights\openai-gpt"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$base = "https://huggingface.co/openai-community/openai-gpt/resolve/main"
$files = "config.json", "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "model.safetensors"
foreach ($f in $files) {
    for ($i = 1; $i -le 100; $i++) {
        Write-Host "[$f] attempt $i"
        curl.exe -L --retry 5 --connect-timeout 30 --speed-time 60 --speed-limit 1000 -C - --progress-bar `
                 -o "$dir\$f" "$base/$f"
        if ($LASTEXITCODE -eq 0) { break }
        Start-Sleep -Seconds 5
    }
}
Get-ChildItem $dir | Format-Table Name, Length
Write-Host "model.safetensors should be ~479 MB (compare with huggingface.co/openai-community/openai-gpt/tree/main)."
Write-Host "A truncated file fails to load; just run this script again and it resumes."
