$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeDirectory = Join-Path $projectRoot '.data\runtime'
$logDirectory = Join-Path $projectRoot '.data\logs'
$pythonPath = Join-Path $projectRoot 'backend\.venv\Scripts\python.exe'
$runtimeTool = Join-Path $PSScriptRoot 'dev_runtime.py'
New-Item -ItemType Directory -Path $runtimeDirectory, $logDirectory -Force | Out-Null

function Enter-LauncherLock {
    $script:launcherLock = New-Object System.Threading.Mutex($false, 'Local\DevFlowAI-DevelopmentLauncher')
    try { $acquired = $script:launcherLock.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $acquired = $true }
    if (-not $acquired) { throw 'Another DevFlow start/stop operation is running. Wait for it to finish.' }
}

function Exit-LauncherLock {
    if ($script:launcherLock) { $script:launcherLock.ReleaseMutex(); $script:launcherLock.Dispose() }
}

function Set-ProjectEnvironment {
    $envFile = Join-Path $projectRoot '.env'
    if (-not (Test-Path -LiteralPath $envFile)) { throw '.env is missing. Restore it from .data/backups; do not replace it with the example.' }
    # Old demo scripts exported terminal overrides. The project .env is authoritative here.
    $names = @('DATABASE_URL','MILVUS_ENABLED','EMBEDDING_PROVIDER','EMBEDDING_MODEL','EMBEDDING_DIMENSIONS',
        'LOCAL_EMBEDDING_DEVICE','LOCAL_EMBEDDING_PRELOAD','AUTO_SYNC_ENABLED','DEMO_MODE','RAG_LLM_ENABLED','RAGAS_ENABLED','PYTHONPATH',
        'COMPOSE_PROJECT_NAME','COMPOSE_FILE','COMPOSE_PROFILES','COMPOSE_ENV_FILES')
    foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') { $names += $Matches[1] }
    }
    # Remove the variable rather than setting null: recent PowerShell/.NET
    # versions can preserve an empty value, which overrides valid .env values.
    foreach ($name in ($names | Select-Object -Unique)) { Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue }
    $env:PYTHONUTF8 = '1'
    $env:NO_PROXY = (@($env:NO_PROXY, 'localhost', '127.0.0.1', '::1') | Where-Object { $_ }) -join ','
}

function Invoke-RuntimeCheck {
    param([string[]]$CheckArguments)
    $output = & $pythonPath $runtimeTool @CheckArguments
    if ($LASTEXITCODE -ne 0) { throw "Runtime check failed: $($CheckArguments -join ' '). See the error above." }
    return ($output | ConvertFrom-Json)
}

function Get-ServiceRecord {
    param([string]$Name)
    $file = Join-Path $runtimeDirectory "$Name-process.json"
    if (-not (Test-Path -LiteralPath $file)) { return $null }
    return (Get-Content -LiteralPath $file -Raw -Encoding UTF8 | ConvertFrom-Json)
}

function Get-RecordedProcess {
    param($Record)
    if (-not $Record) { return $null }
    $process = Get-Process -Id $Record.id -ErrorAction SilentlyContinue
    if ($process -and $process.StartTime.ToUniversalTime() -eq ([datetime]$Record.started_at).ToUniversalTime()) { return $process }
    return $null
}

function Invoke-ProjectCompose {
    param([string[]]$ComposeArguments)
    & docker compose --project-name devflow-ai --project-directory $projectRoot --env-file (Join-Path $projectRoot '.env') `
        -f (Join-Path $projectRoot 'docker-compose.yml') @ComposeArguments
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose failed. Check Docker Desktop and the output above.' }
}

function Test-DockerReady {
    # Windows PowerShell can turn native stderr into terminating errors. A stopped engine is expected here.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & docker info --format '{{.ServerVersion}}' 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    }
    finally { $ErrorActionPreference = $previousPreference }
}
