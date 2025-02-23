# Set working directory
Set-Location "d:\projects\gmail_reader"

# Setup logging
$logFile = ".\logs\gmail_reader_$(Get-Date -Format 'yyyy-MM-dd').log"
$ErrorActionPreference = "Continue"

# Create logs directory if it doesn't exist
if (!(Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs"
}

# Run the Python script and capture output
try {
    $output = & python main.py 2>&1
    $output | Out-File -Append $logFile
    Write-Output "Script executed successfully. Check logs at: $logFile"
} catch {
    $_ | Out-File -Append $logFile
    Write-Error "Error occurred while running the script. Check logs at: $logFile"
    exit 1
}