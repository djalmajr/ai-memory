function Get-AiMemoryCwd {
    param([string] $Payload)
    if (-not $Payload) { return $null }
    try {
        $Parsed = $Payload | ConvertFrom-Json -ErrorAction Stop
        foreach ($Name in @("cwd", "current_dir", "working_dir", "directory")) {
            $Value = $Parsed.$Name
            if ($Value -is [string] -and $Value.Length -gt 0) { return $Value }
        }
        $Paths = $Parsed.workspacePaths
        if ($null -ne $Paths -and $Paths.Count -gt 0 -and $Paths[0] -is [string] -and $Paths[0].Length -gt 0) {
            return $Paths[0]
        }
    } catch {
    }
    $match = [regex]::Match($Payload, '"cwd"\s*:\s*"([^"]*)"')
    if ($match.Success) { return $match.Groups[1].Value }
    $workspaceMatch = [regex]::Match($Payload, '"workspacePaths"\s*:\s*\[\s*"([^"]*)"')
    if ($workspaceMatch.Success) { return $workspaceMatch.Groups[1].Value }
    return $null
}

function Resolve-AiMemoryCwd {
    param([string] $Payload, [string] $Agent)
    $Cwd = Get-AiMemoryCwd -Payload $Payload
    if ($Cwd) { return $Cwd }
    if ($Agent -eq "devin" -and $env:DEVIN_PROJECT_DIR) { return $env:DEVIN_PROJECT_DIR }
    if ($Agent -eq "devin") {
        try {
            $Location = (Get-Location).Path
            if ($Location) { return $Location }
        } catch {
        }
    }
    return $null
}

function Get-AiMemoryMarkerToml {
    param([string] $Cwd)
    if (-not $Cwd) { return $null }
    $dir = $Cwd
    while ($dir -and (Test-Path $dir)) {
        $candidate = Join-Path $dir ".ai-memory.toml"
        if (Test-Path $candidate -PathType Leaf) { return $candidate }
        if ($env:HOME -and $dir -eq $env:HOME) { return $null }
        if ($env:USERPROFILE -and $dir -eq $env:USERPROFILE) { return $null }
        $parent = Split-Path $dir -Parent
        if (-not $parent -or $parent -eq $dir) { return $null }
        $dir = $parent
    }
    return $null
}

function Get-AiMemoryTomlKey {
    param([string] $File, [string] $Key)
    if (-not (Test-Path $File -PathType Leaf)) { return $null }
    foreach ($line in Get-Content $File) {
        $m = [regex]::Match($line, "^\s*$Key\s*=\s*`"([^`"]*)`"")
        if ($m.Success) { return $m.Groups[1].Value }
    }
    return $null
}

# Resolve the basename of the MAIN git repository root for $Cwd, following the
# worktree commondir pointer so every linked worktree collapses to one stable
# name. Mirrors the POSIX `ai_memory_repo_root_project`: a containerized server
# cannot see the host checkout, so repo-root must be resolved here. Returns
# $null when git is unavailable or $Cwd is not inside a git work tree.
function Get-AiMemoryRepoRootProject {
    param([string] $Cwd)
    if (-not $Cwd) { return $null }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { return $null }
    $inside = (& git -C $Cwd rev-parse --is-inside-work-tree 2>$null)
    if ($inside -ne "true") { return $null }
    $common = (& git -C $Cwd rev-parse --path-format=absolute --git-common-dir 2>$null)
    if (-not $common) { return $null }
    $root = Split-Path $common -Parent
    if (-not $root -or $root -eq [System.IO.Path]::GetPathRoot($root)) { return $null }
    return Split-Path $root -Leaf
}

function Get-AiMemoryMarkerQuery {
    param([string] $Cwd)
    if (-not $Cwd) { return "" }
    $qs = "&cwd=$([uri]::EscapeDataString($Cwd))"
    $ws = $null
    $proj = $null
    $strategy = $null
    $dropSubagent = $null
    $marker = Get-AiMemoryMarkerToml -Cwd $Cwd
    if ($marker) {
        $ws = Get-AiMemoryTomlKey -File $marker -Key "workspace"
        $proj = Get-AiMemoryTomlKey -File $marker -Key "project"
        $strategy = Get-AiMemoryTomlKey -File $marker -Key "project_strategy"
        $dropSubagent = Get-AiMemoryTomlKey -File $marker -Key "drop_subagent_captures"
    }
    # Install-time default baked into the hook command by
    # `install-hooks --project-strategy` fills the strategy only when no marker
    # pinned one. A marker's explicit project / project_strategy still win.
    if (-not $strategy -and $env:AI_MEMORY_PROJECT_STRATEGY) {
        $strategy = $env:AI_MEMORY_PROJECT_STRATEGY
    }
    # repo-root must be resolved host-side (the server may not see this checkout);
    # only when no explicit project is pinned. Explicit project always wins.
    if (-not $proj -and ($strategy -eq "repo-root" -or $strategy -eq "repo_root")) {
        $proj = Get-AiMemoryRepoRootProject -Cwd $Cwd
    }
    if ($ws) { $qs += "&workspace=$([uri]::EscapeDataString($ws))" }
    if ($proj) { $qs += "&project=$([uri]::EscapeDataString($proj))" }
    if ($strategy) { $qs += "&project_strategy=$([uri]::EscapeDataString($strategy))" }
    # Per-project drop_subagent_captures opt-in: forward to the server, which
    # interprets truthiness (1/true/...) and scopes the drop to this project.
    if ($dropSubagent) { $qs += "&drop_subagent=$([uri]::EscapeDataString($dropSubagent))" }
    return $qs
}

function Get-AiMemoryStateDir {
    if ($env:AI_MEMORY_DATA_DIR) { return $env:AI_MEMORY_DATA_DIR }
    if ($env:XDG_DATA_HOME) { return (Join-Path $env:XDG_DATA_HOME "ai-memory") }
    if ($env:LOCALAPPDATA) { return (Join-Path $env:LOCALAPPDATA "ai-memory") }
    if ($env:HOME) { return (Join-Path $env:HOME ".local/share/ai-memory") }
    return ".ai-memory"
}

function Get-AiMemorySessionIdPath {
    param([string] $Agent)
    return (Join-Path (Join-Path (Get-AiMemoryStateDir) "hook-state") "$Agent-session-id")
}

function New-AiMemorySessionId {
    param([string] $Agent)
    return "$Agent-$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())-$PID"
}

function Get-AiMemorySessionIdQuery {
    param([string] $Agent, [string] $Event)
    if ($env:AI_MEMORY_SESSION_ID) {
        return "&session_id=$([uri]::EscapeDataString($env:AI_MEMORY_SESSION_ID))"
    }

    $Path = Get-AiMemorySessionIdPath -Agent $Agent
    $SessionId = $null
    if ($Event -ne "session-start" -and (Test-Path $Path -PathType Leaf)) {
        $SessionId = (Get-Content $Path -TotalCount 1 -ErrorAction SilentlyContinue)
    }
    if (-not $SessionId) {
        $SessionId = New-AiMemorySessionId -Agent $Agent
        $Parent = Split-Path $Path -Parent
        New-Item -ItemType Directory -Force -Path $Parent -ErrorAction SilentlyContinue | Out-Null
        Set-Content -Path $Path -Value $SessionId -NoNewline -ErrorAction SilentlyContinue
    }
    return "&session_id=$([uri]::EscapeDataString($SessionId))"
}

function Clear-AiMemorySessionId {
    param([string] $Agent)
    $Path = Get-AiMemorySessionIdPath -Agent $Agent
    Remove-Item -Force -ErrorAction SilentlyContinue $Path
}

function Read-AiMemoryStdin {
    try {
        if (-not [Console]::IsInputRedirected) { return "" }
        $StdinStream = [Console]::OpenStandardInput()
        $StdinReader = [System.IO.StreamReader]::new($StdinStream, [System.Text.Encoding]::UTF8, $false, 4096)
        $ReadTask = $StdinReader.ReadToEndAsync()
        if ($ReadTask.Wait(2000)) {
            $result = $ReadTask.Result
            $StdinReader.Dispose()
            $StdinStream.Dispose()
            return $result
        }
        $StdinReader.Dispose()
        $StdinStream.Dispose()
    } catch {
    }
    return ""
}

function Invoke-AiMemoryHook {
    param(
        [Parameter(Mandatory = $true)] [string] $Event,
        [Parameter(Mandatory = $true)] [string] $Agent,
        [switch] $FetchHandoff,
        [switch] $AntigravityPreInvocationOutput
    )

    $Server = if ($env:AI_MEMORY_HOOK_URL) { $env:AI_MEMORY_HOOK_URL } else { "http://127.0.0.1:49374" }
    $Payload = Read-AiMemoryStdin
    $Cwd = Resolve-AiMemoryCwd -Payload $Payload -Agent $Agent
    $QS = Get-AiMemoryMarkerQuery -Cwd $Cwd
    $SessionQS = ""
    if ($Agent -eq "devin") {
        $SessionQS = Get-AiMemorySessionIdQuery -Agent $Agent -Event $Event
    }
    $Headers = @{}

    if ($env:AI_MEMORY_AUTH_TOKEN) {
        $Headers["Authorization"] = "Bearer $env:AI_MEMORY_AUTH_TOKEN"
    }

    try {
        Invoke-WebRequest `
            -UseBasicParsing `
            -TimeoutSec 3 `
            -Method Post `
            -Uri "$Server/hook?event=$Event&agent=$Agent$QS$SessionQS" `
            -Headers $Headers `
            -ContentType "application/json" `
            -Body $Payload | Out-Null
    } catch {
    }
    if ($Agent -eq "devin" -and $Event -eq "session-end") {
        Clear-AiMemorySessionId -Agent $Agent
    }

    if ($FetchHandoff) {
        try {
            $Response = Invoke-WebRequest `
                -UseBasicParsing `
                -TimeoutSec 2 `
                -Uri "$Server/handoff?agent=$Agent$QS" `
                -Headers $Headers
            if ($null -ne $Response -and $Response.Content) {
                if ($AntigravityPreInvocationOutput) {
                    $Payload = @{
                        injectSteps = @(@{ ephemeralMessage = $Response.Content })
                    }
                    [Console]::Out.Write(($Payload | ConvertTo-Json -Depth 5 -Compress))
                } else {
                    [Console]::Out.Write($Response.Content)
                }
            } elseif ($AntigravityPreInvocationOutput) {
                [Console]::Out.Write("{}")
            }
        } catch {
            if ($AntigravityPreInvocationOutput) {
                [Console]::Out.Write("{}")
            }
        }
    } elseif ($AntigravityPreInvocationOutput) {
        [Console]::Out.Write("{}")
    }
}
