# Internal implementation for the single root CMD launcher. Importing this module starts no services.
function Start-DevFlow {
    param(
        [switch]$SkipInfrastructure,
        [ValidateRange(1, 65535)][int]$BackendPort = 8001,
        [ValidateRange(1, 65535)][int]$FrontendPort = 3000,
        [switch]$OpenBrowser
    )
    
    . (Join-Path (Split-Path -Parent $PSScriptRoot) 'dev_common.ps1')
    
    function Wait-DockerDesktop {
        if (Test-DockerReady) { return }
        $desktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
        if (-not (Test-Path -LiteralPath $desktop)) { throw 'Docker Desktop is not installed.' }
        Write-Host 'Starting Docker Desktop...'
        Start-Process -FilePath $desktop -WindowStyle Hidden | Out-Null
        $deadline = (Get-Date).AddSeconds(120)
        do {
            Start-Sleep -Seconds 2
            if (Test-DockerReady) { return }
        } while ((Get-Date) -lt $deadline)
        throw 'Docker Desktop did not become ready. Open it and check its status, then run the launcher again.'
    }
    
    function Start-DevService {
        param([string]$Name, [string]$Executable, [string[]]$ServiceArguments, [string]$Directory, [string]$Url)
        $record = Get-ServiceRecord $Name
        $process = Get-RecordedProcess $record
        if ($process) {
            if ($record.url -and $record.url -ne $Url) { throw "$Name is running on another port. Stop the project before changing ports." }
            Write-Host "$Name already running (PID $($process.Id)); checking readiness..."
        }
        else {
            $servicePort = ([Uri]$Url).Port
            if (Get-NetTCPConnection -LocalPort $servicePort -State Listen -ErrorAction SilentlyContinue) { throw "Port $servicePort is used by another process. No process was stopped." }
            $process = Start-Process -FilePath $Executable -ArgumentList $ServiceArguments -WorkingDirectory $Directory `
                -RedirectStandardOutput (Join-Path $logDirectory "$Name.out.log") `
                -RedirectStandardError (Join-Path $logDirectory "$Name.err.log") -WindowStyle Hidden -PassThru
            @{ id = $process.Id; started_at = $process.StartTime.ToUniversalTime().ToString('o'); url = $Url } |
                ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeDirectory "$Name-process.json") -Encoding UTF8
        }
        $deadline = (Get-Date).AddSeconds(180)
        do {
            $process.Refresh()
            if ($process.HasExited) { throw "$Name exited. See .data/logs/$Name.err.log." }
            $response = $null
            try { $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3 }
            catch { Start-Sleep -Milliseconds 700 }
            if ($response) {
                if ($Name -eq 'backend') {
                    $health = $response.Content | ConvertFrom-Json
                    if (-not $health.embedding.loaded -or $health.embedding.device -notlike 'cuda*') { throw 'Backend GPU model is not ready. Stop and restart with this launcher.' }
                    Write-Host "GPU model ready: $($health.embedding.gpu) | $($health.embedding.dtype)"
                }
                Write-Host "$Name ready: $Url (PID $($process.Id))"
                return
            }
        } while ((Get-Date) -lt $deadline)
        throw "$Name did not become ready. See .data/logs/$Name.err.log."
    }
    
    Enter-LauncherLock
    try {
        Set-ProjectEnvironment
        if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'backend/.venv is missing. Restore/install dependencies first.' }
        $nodePath = (Get-Command node.exe -ErrorAction Stop).Source
        if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'frontend\node_modules\next'))) { throw 'Run npm ci in frontend first.' }
        Wait-DockerDesktop
        $preflight = Invoke-RuntimeCheck -CheckArguments @('preflight')
        Write-Host "Original data volumes verified. GPU: $($preflight.gpu) (PyTorch $($preflight.torch))"
        if (-not $SkipInfrastructure) { Invoke-ProjectCompose -ComposeArguments @('up', '-d', '--wait', '--wait-timeout', '120', 'postgres', 'etcd', 'minio', 'milvus') }
        $data = Invoke-RuntimeCheck -CheckArguments @('verify-data')
        Write-Host "Original database verified: $($data.counts.repositories) repositories, $($data.counts.documents) documents."
        if (-not (Get-RecordedProcess (Get-ServiceRecord 'backend'))) {
            $backup = Invoke-RuntimeCheck -CheckArguments @('backup', '--reason', 'start')
            Write-Host "Backup saved: $($backup.backup)"
        }
        $env:NEXT_PUBLIC_API_BASE_URL = "http://127.0.0.1:$BackendPort"
        Write-Host 'Loading the embedding model on GPU; first startup normally takes tens of seconds...'
        Start-DevService -Name 'backend' -Executable $pythonPath `
            -ServiceArguments @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$BackendPort") `
            -Directory (Join-Path $projectRoot 'backend') -Url "http://127.0.0.1:$BackendPort/health"
        Start-DevService -Name 'frontend' -Executable $nodePath `
            -ServiceArguments @('node_modules/next/dist/bin/next', 'dev', '--hostname', '127.0.0.1', '--port', "$FrontendPort") `
            -Directory (Join-Path $projectRoot 'frontend') -Url "http://127.0.0.1:$FrontendPort"
        Write-Host "READY: http://127.0.0.1:$FrontendPort | API: http://127.0.0.1:$BackendPort/docs"
        Write-Host 'Closing this window keeps the project running. Use the stop launcher when finished.'
        if ($OpenBrowser) { Start-Process "http://127.0.0.1:$FrontendPort" }
    }
    finally { Exit-LauncherLock }
}

function Stop-DevFlow {
    . (Join-Path (Split-Path -Parent $PSScriptRoot) 'dev_common.ps1')
    Enter-LauncherLock
    try {
        Set-ProjectEnvironment
        foreach ($name in @('frontend', 'backend')) {
            $record = Get-ServiceRecord $name
            $process = Get-RecordedProcess $record
            if ($process) {
                & taskkill.exe /PID $process.Id /T /F
                if ($LASTEXITCODE -ne 0) { throw "Failed to stop $name." }
            }
            $file = Join-Path $runtimeDirectory "$name-process.json"
            if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file }
        }

        if (-not (Test-DockerReady)) {
            Write-Host 'Docker is not running. App processes are stopped; existing data and backups are preserved.'
            Write-Host 'No new backup was created because the database is unavailable.'
            return
        }
        $postgresRunning = & docker inspect devflow-ai-postgres --format '{{.State.Running}}'
        if ($LASTEXITCODE -ne 0) { throw 'Original PostgreSQL container was not found. Existing data was not removed.' }
        if ($postgresRunning -eq 'true') {
            # Quiesce application writes before taking the backup. A failure here keeps containers running.
            $backup = Invoke-RuntimeCheck -CheckArguments @('backup', '--reason', 'stop')
            Write-Host "Verified backup saved: $($backup.backup)"
        }
        else {
            Write-Host 'Database is already stopped. Existing backups and data are preserved; no new backup was created.'
        }
        Invoke-ProjectCompose -ComposeArguments @('stop', 'milvus', 'minio', 'etcd', 'postgres')
        Write-Host 'STOPPED. History, uploaded files and vector data are preserved.'
    }
    finally { Exit-LauncherLock }
}

Export-ModuleMember -Function Start-DevFlow, Stop-DevFlow
