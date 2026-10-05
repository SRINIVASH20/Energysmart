$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Adding EnergySmart to your Desktop..." -ForegroundColor Cyan
Write-Host ""

# ---- Make sure the app files are actually here ------------------------------------
$batPath = Join-Path $scriptDir "run_app.bat"
$vbsPath = Join-Path $scriptDir "launch_energysmart.vbs"
if (-not (Test-Path $batPath)) {
    Write-Host "Could not find run_app.bat in:" -ForegroundColor Red
    Write-Host "  $scriptDir"
    Write-Host "Run this script from inside the EnergySmart project folder."
    Read-Host "Press Enter to close this window"
    exit 1
}

# ---- Resolve the REAL Desktop folder (handles OneDrive-redirected desktops, etc.) --
$desktop = [Environment]::GetFolderPath("Desktop")
if ([string]::IsNullOrWhiteSpace($desktop) -or -not (Test-Path $desktop)) {
    $desktop = Join-Path $env:USERPROFILE "Desktop"
    if (-not (Test-Path $desktop)) {
        try { New-Item -ItemType Directory -Path $desktop -Force | Out-Null } catch { $desktop = $null }
    }
}
if ([string]::IsNullOrWhiteSpace($desktop) -or -not (Test-Path $desktop)) {
    Write-Host "Could not find or create a Desktop folder on this PC." -ForegroundColor Red
    Read-Host "Press Enter to close this window"
    exit 1
}

# ---- Find the icon (either next to this script, or in an assets subfolder) --------
$iconCandidates = @(
    (Join-Path $scriptDir "assets\energysmart.ico"),
    (Join-Path $scriptDir "energysmart.ico")
)
$iconPath = $iconCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$iconFound = [bool]$iconPath
if (-not $iconFound) {
    Write-Host "Note: couldn't find energysmart.ico. Looked in:" -ForegroundColor Yellow
    foreach ($candidate in $iconCandidates) { Write-Host "  $candidate" }
    Write-Host "The shortcut will still be created, just with a default icon for now."
    Write-Host ""
}

# ---- Decide what to launch: the silent VBS launcher if present, else run_app.bat --
$targetArgs = ""
if (Test-Path $vbsPath) {
    # wscript.exe runs .vbs files with no console window at all (unlike cscript.exe,
    # which some PCs use as the default .vbs handler), so target it explicitly.
    $targetPath = Join-Path $env:WINDIR "System32\wscript.exe"
    $targetArgs = "`"$vbsPath`""
} else {
    Write-Host "Note: launch_energysmart.vbs not found - the shortcut will use run_app.bat" -ForegroundColor Yellow
    Write-Host "instead, which shows a terminal window while the app runs."
    Write-Host ""
    $targetPath = $batPath
}

# ---- Clean up anything left over from earlier attempts ----------------------------
# (an older EnergySmart.vbs dropped straight on the Desktop, or a previous shortcut)
$oldStandaloneVbs = Join-Path $desktop "EnergySmart.vbs"
if (Test-Path $oldStandaloneVbs) {
    Remove-Item $oldStandaloneVbs -Force -ErrorAction SilentlyContinue
}
$shortcutPath = Join-Path $desktop "EnergySmart.lnk"
$hadExistingShortcut = Test-Path $shortcutPath
if ($hadExistingShortcut) {
    Remove-Item $shortcutPath -Force -ErrorAction SilentlyContinue
}

try {
    $ws = New-Object -ComObject WScript.Shell
    $sc = $ws.CreateShortcut($shortcutPath)
    $sc.TargetPath = $targetPath
    if ($targetArgs) {
        $sc.Arguments = $targetArgs
    }
    $sc.WorkingDirectory = $scriptDir
    $sc.WindowStyle = 1
    $sc.Description = "Launch the EnergySmart dashboard"
    if ($iconFound) {
        $sc.IconLocation = "$iconPath,0"
    }
    $sc.Save()

    Write-Host "Done! An `"EnergySmart`" icon was added to your Desktop:" -ForegroundColor Green
    Write-Host "  $shortcutPath"
    Write-Host ""
    Write-Host "Double-click it any time to launch the app - no terminal needed."

    if ($iconFound) {
        # Windows caches shortcut icons aggressively, so even a correctly-set icon
        # can keep showing the old/default picture until the cache is cleared.
        Write-Host ""
        Write-Host "Refreshing Windows' icon cache so the new icon shows up..." -ForegroundColor Cyan
        try {
            Start-Process "ie4uinit.exe" -ArgumentList "-ClearIconCache" -WindowStyle Hidden -Wait -ErrorAction SilentlyContinue
            Stop-Process -Name explorer -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 500
            Start-Process explorer.exe
            Write-Host "Done. If the icon still looks unchanged, sign out and back in once."
        } catch {
            Write-Host "Couldn't refresh the icon cache automatically. If the icon looks" -ForegroundColor Yellow
            Write-Host "unchanged, sign out and back in, or restart the PC, and it will update."
        }
    }
} catch {
    Write-Host "Something went wrong creating the shortcut:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    Write-Host ""
    Write-Host "You can still start the app by double-clicking run_app.bat in this folder."
}

Write-Host ""
Read-Host "Press Enter to close this window"
