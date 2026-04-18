<#
PowerShell helper to start main.py in background and capture logs/pid.
Usage:
  # Use defaults
  .\scripts\run_main_nohup.ps1

  # Override defaults
  $env:PYTHON='D:/anaconda/envs/stdplm/python.exe'; .\scripts\run_main_nohup.ps1 -SampleLen 9 -PredictLen 3
#>
param(
    [string]$Python = $env:PYTHON,
    [string]$Task = 'prediction',
    [string]$Model = 'transformer',
    [int]$LLMLayers = 3,
    [int]$BatchSize = 1,
    [int]$SampleLen = 9,
    [int]$PredictLen = 3,
    [int]$SagTokens = 128,
    [int]$TruncK = 64,
    [int]$Epoch = 10,
    [int]$ValEpoch = 1,
    [switch]$SandglassAttn,
    [switch]$NodeEmbedding,
    [switch]$TimeToken,
    [double]$Dropout = 0.2,
    [string]$LR = '1e-4',
    [string]$WeightDecay = '1e-3',
    [int]$Patience = 5,
    [string]$LogRoot = './logs',
    [string]$PredictVars = 'flow,wind,wave',
    [string]$LogDir = ''
)

if (-not $Python) { $Python = 'python' }

# Work from repository root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $scriptDir '..')

if (-not $LogDir) {
    $ts = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
    $LogDir = Join-Path $LogRoot "${ts}_run"
}
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$stdout = Join-Path $LogDir 'stdout.log'
$stderr = Join-Path $LogDir 'stderr.log'
$pidfile = Join-Path $LogDir 'pid.txt'

# Build argument list
$argsList = @(
    'main.py',
    '--task', $Task,
    '--model', $Model,
    '--llm_layers', $LLMLayers.ToString(),
    '--batch_size', $BatchSize.ToString(),
    '--sample_len', $SampleLen.ToString(),
    '--predict_len', $PredictLen.ToString(),
    '--sag_tokens', $SagTokens.ToString(),
    '--trunc_k', $TruncK.ToString(),
    '--epoch', $Epoch.ToString(),
    '--val_epoch', $ValEpoch.ToString(),
    '--dropout', $Dropout.ToString(),
    '--lr', $LR,
    '--weight_decay', $WeightDecay,
    '--patience', $Patience.ToString(),
    '--log_root', $LogDir,
    '--predict_vars', $PredictVars
)
if ($SandglassAttn) { $argsList += '--sandglassAttn' }
if ($NodeEmbedding) { $argsList += '--node_embedding' }
if ($TimeToken) { $argsList += '--time_token' }

# Start process and redirect output
$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $Python
$startInfo.Arguments = $argsList -join ' '
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $startInfo
$process.Start() | Out-Null

$process.Id | Out-File -FilePath $pidfile -Encoding ascii

# Async copy streams to files
$stdoutStream = [System.IO.File]::Open($stdout, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
$stderrStream = [System.IO.File]::Open($stderr, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
$swOut = New-Object System.IO.StreamWriter($stdoutStream)
$swErr = New-Object System.IO.StreamWriter($stderrStream)

Start-Job -ScriptBlock {
    param($proc, $swOut, $swErr)
    while (-not $proc.HasExited) {
        try { $line = $proc.StandardOutput.ReadLine(); if ($line -ne $null) { $swOut.WriteLine($line); $swOut.Flush() } } catch {}
        try { $eline = $proc.StandardError.ReadLine(); if ($eline -ne $null) { $swErr.WriteLine($eline); $swErr.Flush() } } catch {}
        Start-Sleep -Milliseconds 200
    }
} -ArgumentList $process, $swOut, $swErr | Out-Null

Write-Host "Started PID: $($process.Id)"
Write-Host "Stdout: $stdout"
Write-Host "Stderr: $stderr"
Write-Host "Run dir: $LogDir"
