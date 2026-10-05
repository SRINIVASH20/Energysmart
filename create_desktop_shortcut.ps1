$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Creating a Desktop shortcut for EnergySmart..." -ForegroundColor Cyan
Write-Host ""

# Resolve the REAL Desktop folder. On some PCs (managed/lab machines, OneDrive
# "Known Folder Move", custom setups) %USERPROFILE%\Desktop doesn't exist even
# though a Desktop folder does - this asks Windows where it actually is.
$desktop = [Environment]::GetFolderPath("Desktop")

if ([string]::IsNullOrWhiteSpace($desktop) -or -not (Test-Path $desktop)) {
    $desktop = Join-Path $env:USERPROFILE "Desktop"
    if (-not (Test-Path $desktop)) {
        try {
            New-Item -ItemType Directory -Path $desktop -Force | Out-Null
        } catch {
            $desktop = $null
        }
    }
}

if ([string]::IsNullOrWhiteSpace($desktop) -or -not (Test-Path $desktop)) {
    Write-Host "Could not find or create a Desktop folder on this PC." -ForegroundColor Red
    Write-Host "You can still start the app by double-clicking run_app.bat in this folder."
    Write-Host ""
    Read-Host "Press Enter to close this window"
    exit 1
}

$shortcutPath = Join-Path $desktop "EnergySmart.lnk"
$vbsPath = Join-Path $scriptDir "launch_energysmart.vbs"
$batPath = Join-Path $scriptDir "run_app.bat"
$targetArgs = ""
if (Test-Path $vbsPath) {
    # Run via wscript.exe explicitly (not the system default .vbs handler, which on
    # some PCs is cscript.exe and would show a console) so the app launches silently.
    $targetPath = Join-Path $env:WINDIR "System32\wscript.exe"
    $targetArgs = "`"$vbsPath`""
} else {
    Write-Host "Note: launch_energysmart.vbs not found - the shortcut will use run_app.bat" -ForegroundColor Yellow
    Write-Host "instead, which shows a terminal window while the app runs."
    Write-Host ""
    $targetPath = $batPath
}
$iconCandidates = @(
    (Join-Path $scriptDir "assets\energysmart.ico"),
    (Join-Path $scriptDir "energysmart.ico")
)
$iconPath = $iconCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$iconFound = [bool]$iconPath

if (-not $iconFound) {
    Write-Host "Note: couldn't find energysmart.ico. Looked in:" -ForegroundColor Yellow
    foreach ($candidate in $iconCandidates) {
        Write-Host "  $candidate"
    }
    Write-Host "Put energysmart.ico in this folder (or its 'assets' subfolder) and the"
    Write-Host "shortcut will pick it up next time you run this script."
    Write-Host ""
}

# If a shortcut already exists (e.g. from an earlier run before the icon was added),
# delete it first. Editing an existing .lnk in place doesn't always make Windows
# redraw its icon; recreating the file from scratch does.
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

    Write-Host "Done! An `"EnergySmart`" shortcut was added to your Desktop:" -ForegroundColor Green
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
